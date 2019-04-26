from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import random


def RandomCrop(img, jitter=0.3, target_shape=None):
    oh = img.height
    ow = img.width

    dw = int(ow * jitter)
    dh = int(oh * jitter)

    pleft = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    pbot = random.randint(-dh, dh)

    swidth = ow - pleft - pright        # [ow-2*jitter*ow, ow+2*jitter*ow]
    sheight = oh - ptop - pbot          # [oh-2*jitter*oh, oh+2*jitter*oh]
    # pleft + swidth - 1                  [ow-3*jitter*ow+1, ow+3*jitter*ow+1]
    # ptop + sheight - 1                  [oh-3*jitter*oh+1, oh+3*jitter*oh+1]
    cropped = img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))
    if not target_shape:
        img = cropped.resize((ow, oh))
    else:
        img = cropped.resize(target_shape)
    return img


def Flipping(img, vertical=0, horizontal=0, random_flip=False):
    if random_flip:
        rand_hor = random.randint(1, 10000) % 2
        rand_ver = random.randint(1, 10000) % 2
        if rand_hor:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random_flip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        if vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def HSV(img, hue=None, sat=None, val=None):

    def random_scale(x):
        x = random.uniform(1, x)
        if random.randint(1, 10000) % 2 == 0:
            return x
        return 1.0 / x

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    img = img.convert('HSV')
    cs = list(img.split())

    if hue:
        hue = random.uniform(1, hue)
        cs[0] = cs[0].point(change_hue)
    if sat:
        sat = random_scale(sat)
        cs[1] = cs[1].point(lambda i: i * sat)
    if val:
        val = random_scale(val)
        cs[2] = cs[2].point(lambda i: i * val)

    img = Image.merge(img.mode, tuple(cs))
    img = img.convert('RGB')
    return img


def Translation(img):
    pass


def main():
    img = Image.open('tong.jpg').convert('RGB')
    plt.figure(num='Data Agumentation', figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Before')
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.title('After')
    # img = RandomCrop(img)
    img = HSV(img, 0.1, 1.5, 1.5)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
