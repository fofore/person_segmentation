CUDA_VISIBLE_DEVICES=0 python demo_no_data.py --backbone resnet --lr 0.007 --workers 4 --webcam_num 0\
       	--epochs 50 --batch-size 1 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 1 --dataset pascal --resume ./deeplab-resnet.pth.tar --ft

