# bash scripts/gen_demonstration_dexmv.sh relocate-mug


cd third_party/dexmv-sim/examples

task=${1}

CUDA_VISIBLE_DEVICES=0 python gen_demonstration_expert.py --env_name $task \
                        --num_episodes 10 \
                        --root_dir "../../../3D-Diffusion-Policy/data/" \
                        --expert_ckpt_path "../demonstrations/${task}.pkl" \
                        --img_size 84 \
                        --not_use_multi_view \
                        --use_point_crop
