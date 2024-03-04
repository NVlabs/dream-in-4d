#!/bin/bash


prompts=(
a_cat_shaped_kite_sits_in_the_grass.yaml
corgi.yaml
eagle.yaml
fish.yaml
rabbit_food.yaml
)
for i in $(seq 0 $((${#prompts[@]} - 1)))
do
echo ${i}
echo ~/ngc-cli/ngc batch run --name zeroscope_${prompts[${i}]} --instance dgxa100.80g.1.norm --image "nvidian/lpr/yuzheng:three_studio_mult_arch" --result /results --workspace yufeng_west3:yufeng_west3  --ace nv-us-west-3 --commandline "cd /yufeng_west3/dream-in-4D; pip install av; pip install -e ../MVDream; python launch.py  --config python launch.py --config configs/stage_2/stage2_zeroscope_144x80.yaml configs/stage_2/prompts/${prompts[${i}]}  --train --gpu 0; "
~/ngc-cli/ngc batch run --name zeroscope_${prompts[${i}]} --instance dgxa100.80g.1.norm --image "nvidian/lpr/yuzheng:three_studio_mult_arch" --result /results --workspace yufeng_west3:yufeng_west3  --ace nv-us-west-3 --commandline "cd /yufeng_west3/dream-in-4D; pip install av; pip install -e ../MVDream; python launch.py  --config python launch.py --config configs/stage_2/stage2_zeroscope_144x80.yaml configs/stage_2/prompts/${prompts[${i}]}  --train --gpu 0; "
done

#prompts=(
#a_man_drinking_beer.yaml
#)
#for i in $(seq 0 $((${#prompts[@]} - 1)))
#do
#echo ${i}
#echo ~/ngc-cli/ngc batch run --name zeroscope_72_40_${prompts[${i}]} --instance dgxa100.80g.1.norm --image "nvidian/lpr/yuzheng:three_studio_mult_arch" --result /results --workspace yufeng_west3:yufeng_west3  --ace nv-us-west-3 --commandline "cd /yufeng_west3/dream-in-4D;  cp scheduling_ddim.py /usr/local/lib/python3.8/dist-packages/diffusers/schedulers/scheduling_ddim.py; pip install av; pip install -e ../MVDream; python launch.py  --config python launch.py --config configs/stage_2/stage2_zeroscope_72_40_2.yaml configs/stage_2/prompts/${prompts[${i}]}  --train --gpu 0; "
#~/ngc-cli/ngc batch run --name zeroscope_72_40_${prompts[${i}]} --instance dgxa100.80g.1.norm --image "nvidian/lpr/yuzheng:three_studio_mult_arch" --result /results --workspace yufeng_west3:yufeng_west3  --ace nv-us-west-3 --commandline "cd /yufeng_west3/dream-in-4D;  cp scheduling_ddim.py /usr/local/lib/python3.8/dist-packages/diffusers/schedulers/scheduling_ddim.py; pip install av; pip install -e ../MVDream; python launch.py  --config python launch.py --config configs/stage_2/stage2_zeroscope_72_40_2.yaml configs/stage_2/prompts/${prompts[${i}]}  --train --gpu 0; "
#done

#prompts=(
#a_man_drinking_beer.yaml
#)
#for i in $(seq 0 $((${#prompts[@]} - 1)))
#do
#echo ${i}
#echo ~/ngc-cli/ngc batch run --name modelscope_${prompts[${i}]} --instance dgxa100.80g.1.norm --image "nvidian/lpr/yuzheng:three_studio_mult_arch" --result /results --workspace yufeng_west3:yufeng_west3  --ace nv-us-west-3 --commandline "cd /yufeng_west3/dream-in-4D;  cp scheduling_ddim.py /usr/local/lib/python3.8/dist-packages/diffusers/schedulers/scheduling_ddim.py; pip install av; pip install -e ../MVDream; python launch.py  --config python launch.py --config configs/stage_2/stage2_modelscope_256x256.yaml configs/stage_2/prompts/${prompts[${i}]}  --train --gpu 0; "
#~/ngc-cli/ngc batch run --name modelscope_${prompts[${i}]} --instance dgxa100.80g.1.norm --image "nvidian/lpr/yuzheng:three_studio_mult_arch" --result /results --workspace yufeng_west3:yufeng_west3  --ace nv-us-west-3 --commandline "cd /yufeng_west3/dream-in-4D;  cp scheduling_ddim.py /usr/local/lib/python3.8/dist-packages/diffusers/schedulers/scheduling_ddim.py; pip install av; pip install -e ../MVDream; python launch.py  --config python launch.py --config configs/stage_2/stage2_modelscope_256x256.yaml configs/stage_2/prompts/${prompts[${i}]}  --train --gpu 0; "
#done


#prompts=(
#cat_sing.yaml
#dog_skateboard.yaml
#dog_superhero.yaml
#fish.yaml
#fox_game.yaml
#goat_beer.yaml
#monkey_candybar.yaml
#panda_book.yaml
#panda_icecream.yaml
#squirrel_motorcycle.yaml
#)
#for i in $(seq 0 $((${#prompts[@]} - 1)))
#do
#echo ${i}
#echo ~/ngc-cli/ngc batch run --name text-to-3D-${prompts[${i}]} --instance dgxa100.80g.1.norm --image "nvidian/lpr/yuzheng:three_studio_mult_arch" --result /results --workspace yufeng_west3:yufeng_west3  --ace nv-us-west-3 --commandline "cd /yufeng_west3/dream-in-4D; pip install -e ../MVDream; python launch.py  --config configs/stage_1/mvdream-sd21-sd.yaml configs/stage_1/text-to-3D/${prompts[${i}]} --train --gpu 0; "
#~/ngc-cli/ngc batch run --name text-to-3D-${prompts[${i}]} --instance dgxa100.80g.1.norm --image "nvidian/lpr/yuzheng:three_studio_mult_arch" --result /results --workspace yufeng_west3:yufeng_west3  --ace nv-us-west-3 --commandline "cd /yufeng_west3/dream-in-4D; pip install -e ../MVDream; python launch.py  --config configs/stage_1/mvdream-sd21-sd.yaml configs/stage_1/text-to-3D/${prompts[${i}]} --train --gpu 0; "
#done
#
#prompts=(
#a_cat_shaped_kite_sits_in_the_grass.yaml
##a_large_blue_bird_standing_next_to_a_painting_of_flowers.yaml
#a_picture_of_a_flamingo_scratching_its_neck.yaml
##a_small_green_vase_displays_some_small_yellow_blooms.yaml
##bird_eat.yaml
##cartoon_dragon.yaml
#corgi.yaml
#dragon.yaml
#eagle.yaml
#fish.yaml
##horse.yaml
#rabbit_food.yaml
#)
#for i in $(seq 0 $((${#prompts[@]} - 1)))
#do
#echo ${i}
#echo ~/ngc-cli/ngc batch run --name image-to-3D-${prompts[${i}]} --instance dgxa100.80g.1.norm --image "nvidian/lpr/yuzheng:three_studio_mult_arch" --result /results --workspace yufeng_west3:yufeng_west3  --ace nv-us-west-3 --commandline "cd /yufeng_west3/dream-in-4D; pip install -e ../MVDream; python launch.py  --config configs/stage_1/magic123-coarse-if-new.yaml configs/stage_1/image-to-3D/${prompts[${i}]} --train --gpu 0; "
#~/ngc-cli/ngc batch run --name image-to-3D-${prompts[${i}]} --instance dgxa100.80g.1.norm --image "nvidian/lpr/yuzheng:three_studio_mult_arch" --result /results --workspace yufeng_west3:yufeng_west3  --ace nv-us-west-3 --commandline "cd /yufeng_west3/dream-in-4D; pip install -e ../MVDream; python launch.py  --config configs/stage_1/magic123-coarse-if-new.yaml configs/stage_1/image-to-3D/${prompts[${i}]} --train --gpu 0; "
#done
##
#prompts=(
#A_sks_dog_is_eating_sundae.yaml
#A_sks_dog_is_running.yaml
#A_sks_dog_is_swimming.yaml
#A_sks_dog_is_taking_a_shower.yaml
#superhero_sks_dog_wearing_red_cape_is_flying_through_the_sky.yaml
#)
#for i in $(seq 0 $((${#prompts[@]} - 1)))
#do
#echo ${i}
#echo ~/ngc-cli/ngc batch run --name personalized_3D-${prompts[${i}]} --instance dgxa100.80g.1.norm --image "nvidian/lpr/yuzheng:three_studio_mult_arch" --result /results --workspace yufeng_west3:yufeng_west3  --ace nv-us-west-3 --commandline "cd /yufeng_west3/dream-in-4D; pip install -e ../MVDream; python launch.py  --config configs/stage_1/mvdream-sd21-sd.yaml configs/stage_1/personalized_3D/prompts/${prompts[${i}]} configs/stage_1/personalized_3D/subjects/dog8.yaml --train --gpu 0; "
#~/ngc-cli/ngc batch run --name personalized_3D-${prompts[${i}]} --instance dgxa100.80g.1.norm --image "nvidian/lpr/yuzheng:three_studio_mult_arch" --result /results --workspace yufeng_west3:yufeng_west3  --ace nv-us-west-3 --commandline "cd /yufeng_west3/dream-in-4D; pip install -e ../MVDream; python launch.py  --config configs/stage_1/mvdream-sd21-sd.yaml configs/stage_1/personalized_3D/prompts/${prompts[${i}]} configs/stage_1/personalized_3D/subjects/dog8.yaml --train --gpu 0; "
#done
