#!/bin/bash
## Stage 1
# text-to-3D
python launch.py --config configs/stage_1/mvdream-sd21-sd.yaml configs/stage_1/text-to-3D/dog_superhero.yaml --train --gpu 0 system.SD_view=180

# image-to-3D
python launch.py --config configs/stage_1/magic123-coarse-if-new.yaml configs/stage_1/image-to-3D/corgi.yaml --train --gpu 0  # our implemented version using zero123 and deep-floyd-if guidance. This implementation converges fast.
python launch.py --config configs/magic123-coarse-sd.yaml configs/stage_1/image-to-3D/corgi.yaml --train --gpu 0  # threestudio implementation
python launch.py --config configs/stage_1/magic123-coarse-if.yaml configs/stage_1/image-to-3D/corgi.yaml --train --gpu 0  # threestudio implementation, but with deep-floyd-if guidance

# Personalized 3D
python launch.py --config configs/stage_1/mvdream-sd21-sd.yaml configs/stage_1/personalized_3D/prompts/superhero_sks_dog_wearing_red_cape_is_flying_through_the_sky.yaml configs/stage_1/personalized_3D/subjects/dog8.yaml --train --gpu 0 system.SD_view=180

## Stage 2
python launch.py --config configs/stage_2/stage2_zeroscope_144x80.yaml configs/stage_2/prompts/a_man_drinking_beer.yaml --train --gpu 0