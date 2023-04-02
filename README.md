# Neural Style Transfer (PyTorch)

Paper Reference: https://arxiv.org/pdf/1508.06576.pdf
<p float="left">
  <img src="nst_images/vg_starry_night_mona_lisa_73c3bdb3-329f-4de1-8f31-89eeb0732911.jpg" alt= “” width="270" />
  <img src="nst_images/candy_figures_6729a5f8-1245-4096-a18b-1d03986d2837.jpg" width="270" /> 
  <img src="nst_images/edtaonisl_lion_69e1ec93-56f7-42b6-af32-f54cc4407a79.jpg" width="270" />
</p>


Configuration used:                                                             
- Optimizer Used: L-BFGS
- Content Layer = 'Conv_5'
- Style Layer = 'Conv_1', 'Conv_2', 'Conv_3', 'Conv_4', 'Conv_5'
- CNN = VGG-19 (Pre-trained)
- GPU = Nvidia 1080Ti

# Examples:
|  Style Image | Content Image  |  CW |  SW |  NST Image |
|---|---|---|---|---|
|<img src="./images/style_images/candy.jpg" alt= “” width="250px" height="250px">|<img src="./images/content_images/figures.jpg" alt= “” width="250px" height="250px">|   1|   1000000|  <img src="nst_images/candy_figures_6729a5f8-1245-4096-a18b-1d03986d2837.jpg" alt= “” width="250px" height="250px"> |
|<img src="./images/style_images/ben_giles.jpg" alt= “” width="250px" height="250px">|<img src="./images/content_images/green_bridge.jpeg" alt= “” width="250px" height="250px">|   1|   1000000|  <img src="./nst_images/ben_giles_green_bridge_df5065e9-76ee-4f0d-80c3-3a168901193d.jpg" alt= “” width="250px" height="250px"> |
|<img src="./images/style_images/ben_giles.jpg" alt= “” width="250px" height="250px">|<img src="./images/content_images/green_bridge.jpeg" alt= “” width="250px" height="250px">|   10|   100000|  <img src="nst_images/ben_giles_green_bridge_afff54bc-e067-4914-a38e-c84dccdf3418.jpg" alt= “” width="250px" height="250px"> |
|<img src="./images/style_images/edtaonisl.jpg" alt= “” width="250px" height="250px">|<img src="./images/content_images/green_bridge.jpeg" alt= “” width="250px" height="250px">|   1|   1000000|  <img src="nst_images/edtaonisl_green_bridge_4a5f0651-d986-48cb-8dd6-24a4dc5689a4.jpg" alt= “” width="250px" height="250px"> |
|<img src="./images/style_images/edtaonisl.jpg" alt= “” width="250px" height="250px">|<img src="./images/content_images/lion.jpg" alt= “” width="250px" height="250px">|   1|   1000000|  <img src="nst_images/edtaonisl_lion_69e1ec93-56f7-42b6-af32-f54cc4407a79.jpg" alt= “” width="250px" height="250px"> |

# NST sequence example

The below snapshot is the various stages of NST specifically at a gap of 100 iterations of the LBFGS optimizer from 0 to number of steps.

Eg 1:
 <img src="nst_images/edtaonisl_lion_69e1ec93-56f7-42b6-af32-f54cc4407a79_sequence.jpg" alt= “”  width = "1500px" height="150px">

Eg 2:
  <img src="nst_images/edtaonisl_green_bridge_b56689d8-0344-41c7-a5ae-1586c9270389_sequence.jpg"   width = "1500px" height="150px">

Eg 3:
    <img src="nst_images/candy_figures_6729a5f8-1245-4096-a18b-1d03986d2837_sequence.jpg"   width = "1500px" height="150px">


# How to run this ?

Note above samples are on Nvidia 1080Ti GPU and use L-BFGS optimizer. To use L-BFGS need a GPU machine with CUDA installed. For torch.cuda.is_available() is false use Adam optimizer instead

If starting from noise_img otherwise give the starting image as input.

1. Create a virtual Environment
  `conda create -n env`
2. Activate the environment
  `conda activate env`
3. Install Pytorch
  `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
4. Run
 `python main.py --style 'edtaonisl.jpg' --content 'green_bridge.jpeg' --input "noise" --style_weight 1000000 --content_weight 1 --num_steps 500`
5. To Run using Gradio
   `python gradio_nst`
