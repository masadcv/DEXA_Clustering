import torchvision.transforms.functional as F


class ResizeWithPadding:
    """
    Resize the input image to the given size with padding to preserve aspect ratio.
    Based on: https://github.com/pytorch/vision/issues/6236#issuecomment-1175971587
    Adapted to make it work with current codebase, fixes a couple of logical issues and adds some comments.

    Args:
        w (int): width of the resized image.
        h (int): height of the resized image.
    """

    def __init__(self, w=1024, h=768):
        self.w = w
        self.h = h

    def __call__(self, image):
        h_1, w_1 = image.shape[-2], image.shape[-1]
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1

        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):
            # padding to preserve aspect ratio
            hp = int(w_1 / ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                if hp % 2 == 0:
                    hp_minus = hp // 2
                    hp_plus = hp // 2
                else:
                    hp_minus = hp // 2
                    hp_plus = hp // 2 + 1

                image = F.pad(image, (0, hp_minus, 0, hp_plus), 0, "constant")
                return F.resize(image, [self.h, self.w])

            elif hp < 0 and wp > 0:
                if wp % 2 == 0:
                    wp_minus = wp // 2
                    wp_plus = wp // 2
                else:
                    wp_minus = wp // 2
                    wp_plus = wp // 2 + 1
                image = F.pad(image, (wp_minus, 0, wp_plus, 0), 0, "constant")
                return F.resize(image, [self.h, self.w])

        else:
            return F.resize(image, [self.h, self.w])
