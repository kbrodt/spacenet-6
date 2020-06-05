import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.wkt import loads
from affine import Affine
import rasterio
from rasterio import features


def footprint_mask(df, out_file=None, reference_im=None, geom_col='geometry',
                   do_transform=None, affine_obj=None, shape=(900, 900),
                   out_type='int', burn_value=255, burn_field=None):
    """Convert a dataframe of geometries to a pixel mask.
    Arguments
    ---------
    df : :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame`
        A :class:`pandas.DataFrame` or :class:`geopandas.GeoDataFrame` instance
        with a column containing geometries (identified by `geom_col`). If the
        geometries in `df` are not in pixel coordinates, then `affine` or
        `reference_im` must be passed to provide the transformation to convert.
    out_file : str, optional
        Path to an image file to save the output to. Must be compatible with
        :class:`rasterio.DatasetReader`. If provided, a `reference_im` must be
        provided (for metadata purposes).
    reference_im : :class:`rasterio.DatasetReader` or `str`, optional
        An image to extract necessary coordinate information from: the
        affine transformation matrix, the image extent, etc. If provided,
        `affine_obj` and `shape` are ignored.
    geom_col : str, optional
        The column containing geometries in `df`. Defaults to ``"geometry"``.
    do_transform : bool, optional
        Should the values in `df` be transformed from geospatial coordinates
        to pixel coordinates? Defaults to ``None``, in which case the function
        attempts to infer whether or not a transformation is required based on
        the presence or absence of a CRS in `df`. If ``True``, either
        `reference_im` or `affine_obj` must be provided as a source for the
        the required affine transformation matrix.
    affine_obj : `list` or :class:`affine.Affine`, optional
        Affine transformation to use to convert from geo coordinates to pixel
        space. Only provide this argument if `df` is a
        :class:`geopandas.GeoDataFrame` with coordinates in a georeferenced
        coordinate space. Ignored if `reference_im` is provided.
    shape : tuple, optional
        An ``(x_size, y_size)`` tuple defining the pixel extent of the output
        mask. Ignored if `reference_im` is provided.
    out_type : 'float' or 'int'
    burn_value : `int` or `float`, optional
        The value to use for labeling objects in the mask. Defaults to 255 (the
        max value for ``uint8`` arrays). The mask array will be set to the same
        dtype as `burn_value`. Ignored if `burn_field` is provided.
    burn_field : str, optional
        Name of a column in `df` that provides values for `burn_value` for each
        independent object. If provided, `burn_value` is ignored.
    Returns
    -------
    mask : :class:`numpy.array`
        A pixel mask with 0s for non-object pixels and `burn_value` at object
        pixels. `mask` dtype will coincide with `burn_value`.
    """
    # start with required checks and pre-population of values
    if out_file and not reference_im:
        raise ValueError(
            'If saving output to file, `reference_im` must be provided.')
    df = _check_df_load(df)

    if len(df) == 0 and not out_file:
        return np.zeros(shape=shape, dtype='uint8')

    if do_transform is None:
        # determine whether or not transform should be done
        do_transform = _check_do_transform(df, reference_im, affine_obj)

    df[geom_col] = df[geom_col].apply(_check_geom)  # load in geoms if wkt
    if not do_transform:
        affine_obj = Affine(1, 0, 0, 0, 1, 0)  # identity transform

    if reference_im:
        reference_im = _check_rasterio_im_load(reference_im)
        shape = reference_im.shape
        if do_transform:
            affine_obj = reference_im.transform

    # extract geometries and pair them with burn values
    if burn_field:
        if out_type == 'int':
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('uint8')))
        else:
            feature_list = list(zip(df[geom_col],
                                    df[burn_field].astype('float32')))
    else:
        feature_list = list(zip(df[geom_col], [burn_value]*len(df)))

    if len(df) > 0:
        output_arr = features.rasterize(shapes=feature_list, out_shape=shape,
                                        transform=affine_obj)
    else:
        output_arr = np.zeros(shape=shape, dtype='uint8')
    if out_file:
        meta = reference_im.meta.copy()
        meta.update(count=1)
        if out_type == 'int':
            meta.update(dtype='uint8')
            meta.update(nodata=0)
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(output_arr, indexes=1)

    return output_arr


def _check_df_load(df):
    """Check if `df` is already loaded in, if not, load from file."""
    if isinstance(df, str):
        if df.lower().endswith('json'):
            return _check_gdf_load(df)
        else:
            return pd.read_csv(df)
    elif isinstance(df, pd.DataFrame):
        return df
    else:
        raise ValueError(f"{df} is not an accepted DataFrame format.")
    
    
def _check_do_transform(df, reference_im, affine_obj):
    """Check whether or not a transformation should be performed."""
    try:
        crs = getattr(df, 'crs')
    except AttributeError:
        return False  # if it doesn't have a CRS attribute

    if not crs:
        return False  # return False for do_transform if crs is falsey
    elif crs and (reference_im is not None or affine_obj is not None):
        # if the input has a CRS and another obj was provided for xforming
        return True


def _check_geom(geom):
    """Check if a geometry is loaded in.
    Returns the geometry if it's a shapely geometry object. If it's a wkt
    string or a list of coordinates, convert to a shapely geometry.
    """
    if isinstance(geom, BaseGeometry):
        return geom
    elif isinstance(geom, str):  # assume it's a wkt
        return loads(geom)
    elif isinstance(geom, list) and len(geom) == 2:  # coordinates
        return Point(geom)


def _check_rasterio_im_load(im):
    """Check if `im` is already loaded in; if not, load it in."""
    if isinstance(im, str):
        return rasterio.open(im)
    elif isinstance(im, rasterio.DatasetReader):
        return im
    else:
        raise ValueError(
            "{} is not an accepted image format for rasterio.".format(im))
        

def mask_to_poly_geojson(pred_arr, channel_scaling=None, reference_im=None,
                         output_path=None, output_type='geojson', min_area=40,
                         bg_threshold=0, do_transform=None, simplify=False,
                         tolerance=0.5, **kwargs):
    """Get polygons from an image mask.
    Arguments
    ---------
    pred_arr : :class:`numpy.ndarray`
        A 2D array of integers. Multi-channel masks are not supported, and must
        be simplified before passing to this function. Can also pass an image
        file path here.
    channel_scaling : :class:`list`-like, optional
        If `pred_arr` is a 3D array, this argument defines how each channel
        will be combined to generate a binary output. channel_scaling should
        be a `list`-like of length equal to the number of channels in
        `pred_arr`. The following operation will be performed to convert the
        multi-channel prediction to a 2D output ::
            sum(pred_arr[channel]*channel_scaling[channel])
        If not provided, no scaling will be performend and channels will be
        summed.
    reference_im : str, optional
        The path to a reference geotiff to use for georeferencing the polygons
        in the mask. Required if saving to a GeoJSON (see the ``output_type``
        argument), otherwise only required if ``do_transform=True``.
    output_path : str, optional
        Path to save the output file to. If not provided, no file is saved.
    output_type : ``'csv'`` or ``'geojson'``, optional
        If ``output_path`` is provided, this argument defines what type of file
        will be generated - a CSV (``output_type='csv'``) or a geojson
        (``output_type='geojson'``).
    min_area : int, optional
        The minimum area of a polygon to retain. Filtering is done AFTER
        any coordinate transformation, and therefore will be in destination
        units.
    bg_threshold : int, optional
        The cutoff in ``mask_arr`` that denotes background (non-object).
        Defaults to ``0``.
    simplify : bool, optional
        If ``True``, will use the Douglas-Peucker algorithm to simplify edges,
        saving memory and processing time later. Defaults to ``False``.
    tolerance : float, optional
        The tolerance value to use for simplification with the Douglas-Peucker
        algorithm. Defaults to ``0.5``. Only has an effect if
        ``simplify=True``.
    Returns
    -------
    gdf : :class:`geopandas.GeoDataFrame`
        A GeoDataFrame of polygons.
    """

    mask_arr = preds_to_binary(pred_arr, channel_scaling, bg_threshold)

    if do_transform and reference_im is None:
        raise ValueError(
            'Coordinate transformation requires a reference image.')

    if do_transform:
        with rasterio.open(reference_im) as ref:
            transform = ref.transform
            crs = ref.crs
            ref.close()
    else:
        transform = Affine(1, 0, 0, 0, 1, 0)  # identity transform
        crs = rasterio.crs.CRS()

    mask = mask_arr > bg_threshold
    mask = mask.astype('uint8')

    polygon_generator = features.shapes(mask_arr,
                                        transform=transform,
                                        mask=mask)
    polygons = []
    values = []  # pixel values for the polygon in mask_arr
    for polygon, value in polygon_generator:
        p = shape(polygon).buffer(0.0)
        if p.area >= min_area:
            polygons.append(shape(polygon).buffer(0.0))
            values.append(value)

    polygon_gdf = gpd.GeoDataFrame({'geometry': polygons, 'value': values},
                                   crs=crs.to_wkt())
    if simplify:
        polygon_gdf['geometry'] = polygon_gdf['geometry'].apply(
            lambda x: x.simplify(tolerance=tolerance)
        )
    # save output files
    if output_path is not None:
        if output_type.lower() == 'geojson':
            if len(polygon_gdf) > 0:
                polygon_gdf.to_file(output_path, driver='GeoJSON')
            else:
                save_empty_geojson(output_path, polygon_gdf.crs.to_epsg())
        elif output_type.lower() == 'csv':
            polygon_gdf.to_csv(output_path, index=False)

    return polygon_gdf


def preds_to_binary(pred_arr, channel_scaling=None, bg_threshold=0):
    """Convert a set of predictions from a neural net to a binary mask.
    Arguments
    ---------
    pred_arr : :class:`numpy.ndarray`
        A set of predictions generated by a neural net (generally in ``float``
        dtype). This can be a 2D array or a 3D array, in which case it will
        be convered to a 2D mask output with optional channel scaling (see
        the `channel_scaling` argument). If a filename is provided instead of
        an array, the image will be loaded using scikit-image.
    channel_scaling : `list`-like of `float`s, optional
        If `pred_arr` is a 3D array, this argument defines how each channel
        will be combined to generate a binary output. channel_scaling should
        be a `list`-like of length equal to the number of channels in
        `pred_arr`. The following operation will be performed to convert the
        multi-channel prediction to a 2D output ::
            sum(pred_arr[channel]*channel_scaling[channel])
        If not provided, no scaling will be performend and channels will be
        summed.
    bg_threshold : `int` or `float`, optional
        The cutoff to set to distinguish between background and foreground
        pixels in the final binary mask. Binarization takes place *after*
        channel scaling and summation (if applicable). Defaults to 0.
    Returns
    -------
    mask_arr : :class:`numpy.ndarray`
        A 2D boolean ``numpy`` array with ``True`` for foreground pixels and
        ``False`` for background.
    """
    pred_arr = _check_skimage_im_load(pred_arr).copy()

    if len(pred_arr.shape) == 3:
        if pred_arr.shape[0] < pred_arr.shape[-1]:
            pred_arr = np.moveaxis(pred_arr, 0, -1)
        if channel_scaling is None:  # if scale values weren't provided
            channel_scaling = np.ones(shape=(pred_arr.shape[-1]),
                                      dtype='float')
        pred_arr = np.sum(pred_arr*np.array(channel_scaling), axis=-1)

    mask_arr = (pred_arr > bg_threshold).astype('uint8')

    return mask_arr*255


def _check_skimage_im_load(im):
    """Check if `im` is already loaded in; if not, load it in."""
    if isinstance(im, str):
        return skimage.io.imread(im)
    elif isinstance(im, np.ndarray):
        return im
    else:
        raise ValueError(
            "{} is not an accepted image format for scikit-image.".format(im))