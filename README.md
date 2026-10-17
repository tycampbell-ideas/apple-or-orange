# Apples vs. Oranges Image Classifier

My first image classification model, built while working through fast.ai's
*Practical Deep Learning for Coders* (lessons 1–2). It classifies a photo as an
apple or an orange.

## Tools
- **fastai** and **PyTorch**, fine-tuning a pretrained **ResNet-18**
- **Kaggle** notebooks (free GPU) for training
- **Gradio** and **Streamlit** for the web app

## Steps
1. Chose two categories: apples and oranges
2. Wrote a function to search for image URLs with DuckDuckGo
3. Downloaded ~120 images per class, using varied search terms
4. Cleaned the data: removed broken files and exact duplicates
5. Built the training/validation split (80/20) with a fastai `DataBlock`
6. Fine-tuned ResNet-18 using transfer learning
7. Evaluated with a confusion matrix and the highest-loss images
8. Tested on hard cases, found failures, added targeted data, and retrained
9. Built a Gradio demo (on Kaggle) and a Streamlit app (runs locally)

## What I learned: the model was cheating with color
My first model scored 0% error on the validation set, but when I tested it on
harder images, a photo of green, unripe oranges was classified as **100% apple**.
Most of my training images were clean stock photos where apples were red or green
and oranges were orange, so the model had learned a shortcut: *color* rather than
shape or texture.

To fix it, I added images that break that pattern (green/unripe oranges, yellow
apples, fruit on trees) and retrained. The same green-orange photo is now
classified correctly.

Key takeaways:
- **A low validation error doesn't mean the model generalizes.** It's only as
  good as how well the validation data represents real-world images.
- **Confidence isn't correctness.** The model was 100% sure and completely wrong.
- **Improving the data fixed the problem; changing the model wasn't necessary.**

## Limitations
- Small validation set (~50 images), so the true error rate on new photos is
  uncertain.
- Trained mostly on web/stock images; it may struggle with unusual lighting,
  cut fruit, or mixed citrus.

## Next steps
- Public deployment (planned after learning Docker)
- Test on a larger set of my own phone photos
