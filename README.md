# SpaceNet 6

Approach to [SpaceNet 6](https://www.topcoder.com/challenges/30116975?tab=details) challenge on instance segmentation.
The pipeline follows [Open Cities AI Challenge: Segmenting Buildings for Disaster Resilience](https://github.com/kbrodt/building-segmentation-disaster-resilience).

[15nd place](https://www.topcoder.com/challenges/30116975?tab=submissions)
out of 94 on public with 38.70937 jaccard index (top 1 -- 46.5162). Organizers decided to check on private test only top 10 participants on public.
Here is an [announcing the winners](https://medium.com/the-downlinq/spacenet-6-announcing-the-winners-df817712b515).  

### Prerequisites

- GPU(s) with 16Gb RAM
- [NVIDIA apex](https://github.com/NVIDIA/apex)

### Usage

The submission format satisfies the requirements as is in the [submission template](https://github.com/topcoderinc/marathon-docker-template/tree/master/data-plus-code-style).

### Approach

#### Summary
 
- Unet-like architecture with heavy encoder `efficientnet-b7`
- Train on 512x512 crops with 4-channel SAR input
- 5 random folds (both train and test sets share the same Rotterdam city)
- [Multi-channel masks: borders and contacts](https://miro.medium.com/max/1324/1*qePAM_bo6hwzSyOjaekS6g.png)
- Binary-cross-entropy loss
- Predictions are made as `mask * (1 - contact) > 0.45`. It boosted score by 2 points over simple `mask > 0.5`
