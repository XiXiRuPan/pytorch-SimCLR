from torchvision import transforms

def get_transform(image_shape):

    height, width = image_shape[:2]
    transform  = transforms.Compose(
                    [transforms.Resize((height, width)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ])

    return transform