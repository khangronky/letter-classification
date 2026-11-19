# Proposed Model Architectures

This document proposes four Keras models for the nine-class pattern classification task. These are planned configurations, not experimental results or a final model selection. All models accept images of shape **32 × 32**, with `float32` pixel values in **[0, 1]**, and produce nine class probabilities in the order `a, b, c, d, e, f, g, h, k`. Use the same training and validation splits for comparisons, then evaluate the selected configuration once on the local holdout.

Parameter counts below apply to the exact designs specified here. They include biases where stated, batch-normalisation trainable parameters and moving statistics; dropout, reshaping, pooling and activations add no parameters. Actual counts must be checked in Keras after implementation. Input augmentation is excluded from these counts.

| Model | Total parameters | Trainable parameters | Purpose |
|---|---:|---:|---|
| MLP | 296,457 | 296,457 | Dense baseline |
| Custom CNN | 93,833 | 93,833 | Compact spatial-feature model |
| ResNet18, adapted for small images | 11,181,897 | 11,172,297 | Deeper residual model trained from scratch |
| MobileNetV2, frozen base | 2,269,513 | 11,529 | Pretrained feature extractor and new classifier |
| MobileNetV2, entire base trainable | 2,269,513 | 2,235,401 | Upper bound on trainable parameters during fine-tuning |

The MobileNetV2 count changes depending on which layers are unfrozen. Counts alone do not establish inference speed; measure it using the same hardware, batch size and procedure for each model.

## 1. MLP baseline

| Layer | Configuration | Output shape | Parameters |
|---|---|---|---:|
| Input | Grayscale image | 32 × 32 | 0 |
| Flatten | Convert image to a vector | 1,024 | 0 |
| Dense | 256 units, ReLU, bias | 256 | 262,400 |
| Dropout | Rate 0.30 | 256 | 0 |
| Dense | 128 units, ReLU, bias | 128 | 32,896 |
| Dropout | Rate 0.30 | 128 | 0 |
| Dense | 9 units, softmax, bias | 9 | 1,161 |

**Why choose it:** This simple baseline tests whether a network can classify the patterns from pixel values without convolutional layers. Its dense connections can combine information across the image, but flattening discards the explicit neighbourhood structure and introduces many parameters. Dropout is a proposed regularisation choice; its benefit must be checked on validation data.

## 2. Custom CNN

| Layer | Configuration | Output shape | Parameters |
|---|---|---|---:|
| Input and reshape | Add one channel | 32 × 32 × 1 | 0 |
| Convolution | 32 filters, 3 × 3, same padding, ReLU, bias | 32 × 32 × 32 | 320 |
| Max pooling | 2 × 2, stride 2 | 16 × 16 × 32 | 0 |
| Convolution | 64 filters, 3 × 3, same padding, ReLU, bias | 16 × 16 × 64 | 18,496 |
| Max pooling | 2 × 2, stride 2 | 8 × 8 × 64 | 0 |
| Convolution | 128 filters, 3 × 3, same padding, ReLU, bias | 8 × 8 × 128 | 73,856 |
| Global average pooling | Average each feature map | 128 | 0 |
| Dropout | Rate 0.25 | 128 | 0 |
| Dense | 9 units, softmax, bias | 9 | 1,161 |

**Why choose it:** Local convolution filters can learn combinations of edges, rings and streaks while sharing weights across the image. Increasing the channel count allows later layers to combine earlier features; global average pooling keeps the classifier small. This model provides an efficiency comparison against both the MLP and deeper networks. Pooling could also discard useful fine detail, so its effect should be assessed experimentally.

## 3. ResNet18 adapted for small images

Train this residual network from scratch. Use the ResNet18 arrangement of four stages with two basic residual blocks per stage, adapted with a 3 × 3 stride-1 stem and no initial max pool to preserve detail in the small input. Each block contains two 3 × 3 convolutions, each followed by batch normalisation; apply ReLU after the first normalisation and after adding the shortcut. Use bias-free convolutions and standard batch normalisation with trainable scale and offset plus moving mean and variance. The first block of stages 2–4 downsamples using stride 2 in its first convolution and uses a bias-free 1 × 1 stride-2 projection with batch normalisation on the shortcut. Other shortcuts are identities.

| Stage | Configuration | Output shape |
|---|---|---|
| Input and reshape | Single grayscale channel | 32 × 32 × 1 |
| Stem | 3 × 3 convolution, 64 filters, stride 1, same padding; batch normalisation; ReLU | 32 × 32 × 64 |
| Stage 1 | 2 basic blocks, 64 filters | 32 × 32 × 64 |
| Stage 2 | 2 basic blocks, 128 filters; first block stride 2 | 16 × 16 × 128 |
| Stage 3 | 2 basic blocks, 256 filters; first block stride 2 | 8 × 8 × 256 |
| Stage 4 | 2 basic blocks, 512 filters; first block stride 2 | 4 × 4 × 512 |
| Classifier | Global average pooling; dense 9, softmax, bias | 9 |

**Parameters:** 11,181,897 total: 11,172,297 trainable and 9,600 non-trainable batch-normalisation moving statistics. This is a small-image adaptation, not the original ImageNet stem. The 18 refers to the stem convolution, 16 block convolutions and classifier; shortcut projections are additional operations.

**Why choose it:** Residual shortcuts give the network a way to learn changes to features across a deeper stack. This tests whether greater depth helps distinguish subtle patterns. Its large parameter count and inference cost relative to the compact CNN may outweigh any accuracy gain, so compare both measures. Implement it using Keras with the TensorFlow backend. The residual design is based on [He et al., Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

## 4. ImageNet-pretrained MobileNetV2 with fine-tuning

| Stage | Configuration | Output shape |
|---|---|---|
| Input and reshape | Add one grayscale channel | 32 × 32 × 1 |
| Resize and channel conversion | Bilinear resize to 64 × 64; repeat the grayscale channel three times | 64 × 64 × 3 |
| Input scaling | Map the supplied [0, 1] pixels to [-1, 1] | 64 × 64 × 3 |
| Pretrained base | Keras MobileNetV2, ImageNet weights, `alpha=1.0`, `include_top=False` | 2 × 2 × 1,280 |
| Classifier | Global average pooling; dropout 0.20; dense 9, softmax, bias | 9 |

**Parameters:** 2,269,513 total. With the base frozen, the new dense classifier has 11,529 trainable parameters and the remaining 2,257,984 are non-trainable. If the entire base is made trainable, 2,235,401 parameters are trainable and 34,112 batch-normalisation moving-statistic parameters remain non-trainable. A partial fine-tuning stage will fall between these totals; record the actual count for the chosen unfrozen layers.

**Training approach:** First freeze the base and train the new classifier. Then unfreeze the final 20 base layers as an initial fine-tuning experiment, recompile with a lower learning rate, and keep batch-normalisation layers in inference mode. Compare the frozen and fine-tuned variants on the same validation set. The 64 × 64 input is a proposed experiment: resizing increases computation and does not add detail to the original 32 × 32 images. Keep resize, channel conversion and scaling in the final saved model or prediction function if this approach is selected.

**Why choose it:** MobileNetV2 is designed for efficient image recognition and provides a way to test transfer learning against models trained from scratch. Its ImageNet features come from colour photographs, which differ from these grayscale patterns, so transfer learning may not improve accuracy. The pretrained checkpoint, conversion and scaling must be documented; the saved `.keras` model must contain the weights needed for inference. See the [Keras MobileNetV2 documentation](https://keras.io/api/applications/mobilenet/mobilenet_models/) and [TensorFlow transfer-learning guide](https://www.tensorflow.org/guide/keras/transfer_learning).

## Comparison and final choice

Compare validation accuracy, learning curves, trainable and total parameters, and inference time under a documented, consistent measurement procedure. Examine errors and compare augmentation with no augmentation as a separate experiment on the training split. Choose the final configuration based on validation evidence and the accuracy–efficiency trade-off; report the local holdout result only after that choice is fixed. Describe actual results and the final architecture in the report, rather than presenting these proposals as completed experiments.
