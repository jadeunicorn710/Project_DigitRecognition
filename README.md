# CNN Digit Recognition

## Main Directory File Structure

- Python code files: `data_preprocessing.py`, `models.py`, `digits_detection_and_recognition.py`, `run.py`
- `cv_proj.yml`
- **svhnData** folder: 3 subfolders - **Format1**, **Format2**, **test_images**
- `FinalProjectCheckpoints.ipynb`: Google Colab scripts for training & evaluating models and output metrics plots, models, model weights files. Has running logs & results, for information only.
- **metrics_plots** folder: generated metrics plots used in report
- **graded_images** folder: output images, `1.png` - `5.png` are good results, `6.png` - `7.png` are bad results



## Instructions on Running the Code

- Preprocessed train & test datasets (.npy), all three trained models (.json) with corresponding weights (.h5) were already generated and can be directly used to facilitate all tasks.

- You should only need `vgg16_pretrained_model.json` and `vgg16_pretrained_weights.h5` files to get the output images needed. You can download them from the following link, just place them in the main directory.

  https://www.dropbox.com/sh/sqegtw4108ouzlv/AAAqkrvO96vBLhXl2F-6DcU4a?dl=0

- Then, to run the code, simply run the `run.py` in the main directory, *Pycharm perferred*, if you run it in terminal, no arguments are needed. It will directly save all output images in the **graded_images** folder for you. It should take less than a minute.



## Additional Saved Files for Your Reference

Feel free to download any in case your are interested in testing anything additional. Place in main directory unless noted otherwise.

- Saved model and weights files for vgg16_scratch model:

  https://www.dropbox.com/sh/wzbn7o4cl9o00wi/AABPEyW2qgx3xn7qCj1Pl12Fa?dl=0

- Saved model and weights files for customized CNN model:

  https://www.dropbox.com/sh/2tus20ql7wpm7rv/AAC1mcgf1hC0K_SaRRJQyzrva?dl=0

- Saved preprocessed train set and test set (used for model training). Place in **main folder/svhnData/Format2**.

  https://www.dropbox.com/sh/qc5531pby4jyjeh/AADTYtN1X4kcM55S-nGAr_S5a?dl=0
