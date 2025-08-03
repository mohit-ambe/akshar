# the following libraries are allowed since they are builtin to Python
import os
from math import sin, cos, atan2, radians
from random import randrange, seed
from subprocess import call

from Matrix import Matrix

gray_max = 255


def main():
    # get labels and ImageMagick's file location
    # labels convert folder names to their respective character names
    # magick is used to convert pngs and jpgs into the more accessible pgm format
    MAGICK_EXE = [s for s in os.environ['PATH'].split(";") if "Magick" in s][0] + "\\magick.exe"
    with open(os.curdir + "\\datasets\\labels.csv", 'rb') as f:
        LABELS = [line.strip().decode('utf-8') for line in f.readlines() if line]
        LABELS = [l.split(",") for l in LABELS if l]
        LABELS = list(map(list, zip(*LABELS)))

    # source and destination
    dataset_dir = os.curdir  #
    results_dir = os.curdir  #
    extension = ".png"  # or ".jpg"

    characters = [n for n in os.listdir(dataset_dir)]
    sample_size = 25

    for idx, chars in enumerate(characters):
        id = 1
        symbol = LABELS[1][1 + int(chars.split("_")[-1])]

        # first convert an image to pgm format,
        # then generate its samples.
        # delete the converted image after
        for image in os.listdir(dataset_dir + "\\" + chars):
            image = image.replace(extension, "")
            path = dataset_dir + "\\" + chars + "\\" + image
            # print(idx, symbol, image, id)
            call(["magick", path + extension, results_dir + "\\" + image + ".pgm"], executable=MAGICK_EXE)
            generate_samples(symbol, results_dir + "\\" + image + ".pgm", ids=range(id, sample_size + id))
            os.remove(results_dir + "\\" + image + ".pgm")
            id += sample_size


def generate_samples(label, dir, ids=(1,), random_seed=None):
    seeds = set()
    for gen_id in ids:
        # all samples of an image must have a unique seed
        IMAGE_SIZE = 30
        if SEED := random_seed:
            pass
        else:
            SEED = randrange(int(1e4), int(1e5))
            while SEED in seeds:
                SEED = randrange(int(1e4), int(1e5))
            seeds.add(SEED)

        seed(SEED)

        image = read_image(dir)

        # invert if needed
        # image = invert(image)

        image = pad(image, 10)

        # 1: Rotate the image randomly
        for _ in range(randrange(1, 3)):
            image = scale_rotate(image, randrange(-20, 20))

        # 1: Crop and scale image proportionally
        # final size should be approx. IMAGE_SIZE
        image = crop(image)
        if image.rows > IMAGE_SIZE or image.cols > IMAGE_SIZE:
            image = scale_rotate(image, 0, (IMAGE_SIZE - 5) / max(image.rows, image.cols))

        # 2: crop and square the image
        # pad again to meet IMAGE_SIZE
        image = crop(image)
        image = square(image)
        image = pad(image, (IMAGE_SIZE - image.rows) // 2)

        pady = 2 + (IMAGE_SIZE - image.rows) // 2
        padx = 2 + (IMAGE_SIZE - image.cols) // 2

        # 3: Pan the image randomly
        if padx and pady:
            image = pan(image, randrange(-padx, padx), randrange(-pady, pady))

        # edge case where some images are of size IMAGE_SIZE - 1
        if image.rows == image.cols == IMAGE_SIZE - 1:
            pixels = image[:]
            pixels.append([0 for _ in range(image.rows)])
            pixels = [[*p, 0] for p in pixels]
            image = Matrix(pixels)

        # export the sample with its id, label, and used seed
        path = "" + f'{gen_id:05}_{label}_{SEED}'
        write_image(image, path + ".pgm")


def read_image(dir):
    # PGM FORMAT                    EXAMPLE

    # magic number                  P5
    # width height                  3 3
    # maxgrayvalue                  255
    # ............                  0 0 0
    # ...pixels...                  0 0 0
    # ............                  0 0 0

    with open(dir, 'rb') as f:
        raw_data = f.read()
    data = raw_data.split(b'\n')
    read_line = lambda j:"".join([chr(s) for s in data[j]])

    width, height = map(int, read_line(1).split())

    pixels = [[]]

    for i, b in enumerate(raw_data.split(b'\n')[3:]):
        for byte in b:
            pixels[-1].append(byte)
            if len(pixels[-1]) == width and len(pixels) != height:
                pixels.append([])

    pixels[-1].extend([0 for _ in range(width - len(pixels[-1]))])

    return Matrix(pixels)


# pad the image's rows, columns, or both
def pad(mtrx, i, method="both"):
    if i < 0:
        return mtrx
    dr, dc = (0 if method == 'col' else i), (0 if method == 'row' else i)
    m = Matrix(mtrx.rows + dr * 2, mtrx.cols + dc * 2)
    m.rounding(mtrx.floating_digits)
    for r in range(mtrx.rows):
        for c in range(mtrx.cols):
            m[r + dr][c + dc] = mtrx[r][c]
    return m


# use polar coordinates to scale and
# rotate the image about its center
def scale_rotate(mtrx: Matrix, phi: float = 0, p: float = 1):
    phi = radians(phi)
    origin = (mtrx.rows // 2, mtrx.cols // 2)
    dist = lambda a, b:(a ** 2 + b ** 2) ** 0.5
    m = Matrix(mtrx.rows, mtrx.cols)
    m.rounding(mtrx.floating_digits)
    for dr in range(m.rows):
        for dc in range(m.cols):
            # find coords relative to origin
            x, y = dr - origin[0], dc - origin[1]
            # convert to polar coords
            rad = dist(x, y)
            theta = atan2(y, x)
            # transform coords with offset phi
            rotr, rotc = (rad * cos(theta - phi), rad * sin(theta - phi)) if phi else (x, y)
            rotr, rotc = origin[0] + int(rotr / p), origin[1] + int(rotc / p)
            if 0 <= rotr < mtrx.rows and 0 <= rotc < mtrx.cols:
                m[dr][dc] = mtrx[rotr][rotc]
    return m


# image must have black background!
# removes excess padding
def crop(mtrx: Matrix):
    col_range = [mtrx.cols, 0]
    row_range = [mtrx.rows, 0]
    for r in range(mtrx.rows):
        if any(mtrx.row(r)):
            ra, rb = row_range
            row_range = [min(ra, r), max(rb, r)]
    for c in range(mtrx.cols):
        if any(mtrx.col(c)):
            ca, cb = col_range
            col_range = [min(ca, c), max(cb, c)]

    if (row_range[1] - row_range[0]) % 2 != (col_range[1] - col_range[0]) % 2:
        row_range[1] += 1

    m = Matrix(row_range[1] - row_range[0], col_range[1] - col_range[0])
    m.rounding(mtrx.floating_digits)
    for dr in range(m.rows):
        for dc in range(m.cols):
            m[dr][dc] = mtrx[dr + row_range[0]][dc + col_range[0]]
    return m


# add padding until the image is square
def square(mtrx):
    if mtrx.cols < mtrx.rows:
        return pad(mtrx, (mtrx.rows - mtrx.cols) // 2, 'col')
    elif mtrx.cols > mtrx.rows:
        return pad(mtrx, (mtrx.cols - mtrx.rows) // 2, 'row')
    return mtrx


# invert the colors
def invert(mtrx):
    m = Matrix(mtrx.rows, mtrx.cols)
    m.rounding(mtrx.floating_digits)
    for r in range(mtrx.rows):
        for c in range(mtrx.cols):
            m[r][c] = gray_max - mtrx[r][c]
    return m


# shift the image horizontally or vertically
def pan(mtrx, dx, dy):
    m = Matrix(mtrx.rows, mtrx.cols)
    for r in range(m.rows):
        for c in range(m.cols):
            if 0 <= r - dy < mtrx.rows and 0 <= c - dx < mtrx.cols:
                if p := mtrx[r - dy][c - dx]:
                    m[r][c] = p
    return m


def write_image(img: Matrix, dir: str):
    # PGM FORMAT                    EXAMPLE

    # magic number                  P5
    # width height                  3 3
    # maxgrayvalue                  255
    # ............                  0 0 0
    # ...pixels...                  0 0 0
    # ............                  0 0 0

    global gray_max
    with open(dir, 'wb') as f:
        header = f"P5\n{img.cols} {img.rows}\n{gray_max}\n"
        f.write(header.encode('utf-8'))

        # Write the pixel data
        for row in img[:]:
            f.write(bytearray(row))
    return


# add random noise to the image
def noise(mtrx):
    m = Matrix(mtrx[:])
    for r in range(0, mtrx.rows):
        for c in range(0, mtrx.cols):
            m[r][c] += randrange(-100, 100) // 7
            m[r][c] = min(m[r][c], 255)
            m[r][c] = max(m[r][c], 0)
    return m


if __name__ == '__main__':
    main()
