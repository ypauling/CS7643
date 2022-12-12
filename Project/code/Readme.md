# Image2recipe model description

## Directory Structure

project/
    data/
        /images
            /val
            /train
            /test
        /text
            vocab.bin
        /train_lmdb
        /val_lmdb
        /test_lmdb
    /scripts
        params.py
        data_loader.py
        recipe_model.py
        train.py

## Model Input and Output

### Input
There are two parts in the input:

- The INGREDIENTS and RECIPE for each image
- The IMAGE

In data_loader.py, the each sample is returned as a list of five components:

- The food image, a torch tensor.
- The ingredients, coded as a list of integers, used to retreive word2vec trained embedding vectors
- The number of ingredients, or the length of the LSTM sequence.
- The instructions, coded already as embedding using the skip-thought vectors (refer to the paper for details).
- The number of instruction sentences, or the length of the LSTM sequence.

The target, or y, is returned as a list of five components
- 1 or -1, as indicator of whether the image and the food recipe match (remember that training an embedding model is to minimize distance between pairs that are similar and to maximize distance between things that are different). 
- The class index for the image defined in food101 dataset.  See paper for details.
- The class index for the recipe defined in food101 dataset.
- The dataset id for the image.
- The dataset id for the recipe.

Note that element 2 and 3 are used for semantic regularization.

### Output

Output of the model contains four components

- The embedding for the image
- The embedding for the recipe
- The predicted food101 class scores for img embedding
- The predicted food101 class scores for recipe embedding

### Dimensions

- image: [B x C x H x W]
- ingredients: [B x SEQ_LEN]
- n_ingr: [B]
- instructions: [B x SEQ_LEN x LEN_STVEC] STVEC: skip-thought vectors
- n_inst: [B]

- image embedding: [B x EMB_SIZE]
- recipe embedding: [B x EMB_SIZE]

## Things need to be changed
In recipe_model.py: line 86-88.

This is the image model used.  Currently it is resnet50.  Change to some other visual networks for comparison.