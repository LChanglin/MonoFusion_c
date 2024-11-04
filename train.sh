
# # --load_flow_param \
OMP_NUM_THREADS=8 torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
--resume ./pretrained/unisma_all.pth.tar \
--save ./pretrained/unisma_all_kitti.pth.tar \
--padding_factor 32 \
--upsample_factor 4 \
--num_scales 2 \
--attn_splits_list 2 8 \
--corr_radius_list -1 4 \
--prop_radius_list -1 1 \
--lr 2e-5 \
--epoch 8 \
--batch_size 1 \
--stage 'kitti' \
--image_size 384 768 \
--parallel \
# # --load_opt
# # --load_flow_param \


# OMP_NUM_THREADS=8 torchrun --standalone --nnodes=1 --nproc_per_node=4 train_mix.py \
# --resume ./pretrained/gmf9y9y.pth.tar \
# --save ./pretrained/unisma_kitti.pth.tar \
# --padding_factor 32 \
# --upsample_factor 4 \
# --num_scales 2 \
# --attn_splits_list 2 8 \
# --corr_radius_list -1 4 \
# --prop_radius_list -1 1 \
# --lr 4e-5 \
# --epoch 15 \
# --batch_size 1 \
# --stage 'kitti' \
# --image_size 320 1152 \
# --parallel \
# # # --load_opt


# --load_flow_param \

# --batch_size 4 \
# --stage 'driving' \
# --image_size 320 640 \

# --batch_size 3 \
# --stage 'kitti' \
# --image_size 288 960 \