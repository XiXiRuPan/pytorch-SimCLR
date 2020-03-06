import numpy as np
import torch
import augmentations

class Trainer:
    """A Trainer Class used for training"""

    def __init__(self,
        dataloaders=None,
        net=None,
        criterion=None,
        optimizer=None,
        num_epochs=10,
        use_gpu=True
    ):

        self.train_loader, self.test_loader = dataloaders
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.use_gpu = use_gpu

    def transform_image(self, image, num_transforms=2):
        """Function to apply set of augmentations to 'image'
        Args:
            image(np.array):  image to transform.
        Returns:
            transformed_images(np.array): transformed images.
        """

        transformed_images = []
        target_transforms = np.random.choice(augmentations.augmentation_set, size=num_transforms)
        for transform_name in target_transforms:
            transform = getattr(augmentations, transform_name)
            transformed_images.append(transform(image))

        return transformed_images


    def transform_batch(self, raw_images):
        """Function to transform/augment images in a batch"""

        transformed_batch = []
        for image in enumerate(raw_images):
            transformed_images = self.transform_image(image)
            transformed_batch += transformed_images

        transformed_batch = torch.stack(transformed_batch)

        return transformed_batch


    def fit(self):
        """Function to start training 'net'"""

        if self.use_gpu:
            self.net.cuda()
        self.net.train()

        for epoch_idx in range(self.num_epochs):
            for step_idx, batch in enumerate(self.train_loader):

                raw_images, batch_images = batch
                # for every image 'i' in the batch, generate two transformed versions of 'i'
                transformed_batch = self.transform_batch(raw_images)

                if self.use_gpu:
                    transformed_batch = transformed_batch.cuda()
                    batch_images = batch_images.cuda()

                # forward pass original and transformed images
                batch_embeddings = self.net(batch_images)
                transformed_batch_embeddings = self.net(transformed_batch)

                # find loss for original and transformed image
                # TODO