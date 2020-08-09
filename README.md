# Photo-to-Emoji Transformation with TraVeLGAN and Perceptual Loss

Pytorch implementation of Thesis project entitled *"Photo-to-Emoji Transformation with TraVeLGAN and Perceptual Loss"* (or in Chinese, "基於TraVeLGAN與Perceptual Loss實現照⽚轉換表情符號之應⽤")

## Getting Started (Training)
Steps:
1. Download all of the files and folders in this repo and prepare the dataset. In my project, in this project we used [CelebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and Bitmoji dataset run `python create_emojis.py` and set the number of bitmoji images on the `num_emojis` variable.

2. Put the training CelebA dataset inside `dataset/CelebA/train/` folder, and test CelebA dataset inside `dataset/CelebA/test`.

3. Put all the Bitmoji dataset inside `dataset/Bitmoji` folder.

4. Set up the config file inside `configs/cifar.json`. Generally, You can determine the number of epochs, n_save_steps, and batch_size. I use `batch_size=32` for faster converged.

5. Run program using command 
```
python train.py --log log_photo2emoji --project_name photo2emoji  
```


## Testing
Steps:
1. Change the `saved_model` key in `config.json` to be `./log_photo_to_emoji/model_500.pt` or whenever number of iteration model you use.

2. run program using command
```
python testAtoB.py --project_name photo2emoji
```

## Result Samples
![1](./samples/1.png)![2](./samples/2.png)![3](./samples/3.png)![4](./samples/4.png)![5](./samples/5.png)

![6](./samples/6.png)![7](./samples/7.png)![8](./samples/8.png)![9](./samples/9.png)![10](./samples/10.png)

## Comparison
1. TraVeLGAN (Original)

![1](./samples/comparison/TraVeLGAN/1.png)![2](./samples/comparison/TraVeLGAN/2.png)![3](./samples/comparison/TraVeLGAN/3.png)![4](./samples/comparison/TraVeLGAN/4.png)![5](./samples/comparison/TraVeLGAN/5.png)

2. TraVeLGAN + Perceptual Loss

![1](./samples/comparison/TraVeLGAN_with_Perceptual_Loss/1.png)![2](./samples/comparison/TraVeLGAN_with_Perceptual_Loss/2.png)![3](./samples/comparison/TraVeLGAN_with_Perceptual_Loss/3.png)![4](./samples/comparison/TraVeLGAN_with_Perceptual_Loss/4.png)![5](./samples/comparison/TraVeLGAN_with_Perceptual_Loss/5.png)