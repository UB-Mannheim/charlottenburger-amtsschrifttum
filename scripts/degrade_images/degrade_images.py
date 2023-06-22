import imghdr
import random
from itertools import chain
from pathlib import Path

import click
import numpy as np
import ocrodeg
import scipy.ndimage as ndi
from PIL import Image

"""

To-Do:
1. Aktuell werden RGB-PNGs nicht in Grayscale umgewandelt, sondern binarisiert (Zeile 24); habe bisher keinen
passenden Code gefunden, um ein Bild in Graustufen umzuwandeln.
2. Der Code erzeugt bereits ein Bild, das die Papierstruktur nachahmt (Zeile 55). Allerdings klappt die Überlagerung
dieses Bildes mit dem transformierten Bild (in diesem Schritt 'img_blotched') nicht.
 
"""

def thresholded_gauss(vmin, vmax, mu=None, scale=4):
    vmin, vmax = (vmin, vmax) if vmin < vmax else (vmax, vmin)
    if not mu or not (vmin <= mu <= vmax):
        mu = (vmin+vmax)/2
    sigma = abs(vmax-vmin)/scale
    value = random.gauss(mu, sigma)
    while not vmin <= value <= vmax:
        value = random.gauss(mu, sigma)
    return value

@click.command()
@click.argument('imgpaths', nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', type=click.Path(exists=True, path_type=Path), help='Path to font (default: inputfolder)')
@click.option('-d', '--degrade-filters', default=['all'], multiple=True,
              type=click.Choice(['all', 'all-bin', 'rotate', 'transform', 'noise', 'filter', 'binary_blur']),help='Rotate angle')
@click.option('-r', '--rotate', default=(-0.01, 0.01), type=(float, float), help='Rotate angle')
@click.option('-a', '--ansio', default=(0.9, 1.2), type=(float, float), help='Ansio factor Min-Max-Randomvalues')
@click.option('-s', '--scale', default=(0.9, 1.0), type=(float, float),help='Scale factor Min-Max-Randomvalues')
@click.option('-t', '--translation', default=(0.001, 0.03),  type=(float, float), help='Translation Min-Max-Randomvalues')
@click.option('-g', '--gaussian-noise', default=(0.0, 0.5), type=(float, float), help='Gaussian noise')
@click.option('-n', '--flat-noise', default=5.0, type=float, help='1d noise magnitude')
@click.option('-g', '--gaussian-filter', default=(0.1, 2.0), type=(float, float), help='Gaussian filter (blur)')
@click.option('-b', '--binary-blur', default=(0.0, 0.2), type=(float, float), help='Binary blur')
@click.option('--speckle', type=click.Choice(['fg', 'bg', 'all']), help='Add speckles to fg (foreground), bg (background) or all (both)')
@click.option('--speckle-fg-density', default=(0.000001, 0.0001), type=(float, float), help='Density of the fg speckles')
@click.option('--speckle-fg-size', default=(8, 12), type=(float, float), help='Size of the fg speckles')
@click.option('--speckle-fg-roughness', default=2, type=int, help='Roughness of the fg speckles')
@click.option('--speckle-bg-density', default=(0.000001, 0.0001), type=(float, float), help='Density of the bg speckles')
@click.option('--speckle-bg-size', default=(8, 12), type=(float, float), help='Size of the bg speckles')
@click.option('--speckle-bg-roughness', default=2, type=int, help='Roughness of the bg speckles')
@click.option('-p', '--printlike', type=click.Choice(['fibrous', 'multiscale']), help='Add fibrous or multiscale patches (auto adds speckles)')
@click.option('--random-function', default='uniform', type=click.Choice(['uniform', 'gauss', 'random']), help='Select the function go generate the random values' )
@click.option('--gauss-dynamic-mu', is_flag=True, help='Calculate mu as mean of min and max values (default: Use function-individual static mu)')
def degrade(imgpaths, output_dir, degrade_filters, rotate, ansio, scale, translation,  gaussian_noise, flat_noise,
            gaussian_filter, binary_blur, speckle, speckle_fg_density, speckle_fg_size, speckle_fg_roughness,
            speckle_bg_density, speckle_bg_size, speckle_bg_roughness,
            printlike, random_function, gauss_dynamic_mu):
    degrade_filters = ['rotate', 'transform', 'noise', 'filter'] if 'all' in degrade_filters else degrade_filters
    degrade_filters = ['rotate', 'transform', 'noise', 'filter', 'binary_blur'] if 'all-bin' in degrade_filters else degrade_filters
    speckle = ['fg', 'bg'] if speckle and speckle == 'all' else speckle
    imgfiles = chain.from_iterable([[imgpath] if imgpath.is_file() else [filename for
                filename in imgpath.glob('*') if (filename.is_file() and imghdr.what(filename))] for imgpath in imgpaths])
    random_func = {'uniform': random.uniform, 'gauss': thresholded_gauss, 'random': random.random}.get(random_function, random.uniform)
    if random_function == "gauss" and not gauss_dynamic_mu:
        rotate = (*rotate, 0)
        ansio = (*ansio, 1)
        scale = (*scale, 1)
        translation = (*translation, 0)
        gaussian_noise = (*gaussian_noise, 1)
        binary_blur = (*binary_blur, 0.1)
        speckle_fg_density = (*speckle_fg_density, 0.00005)
        speckle_bg_density = (*speckle_bg_density, 0.00005)
        speckle_fg_size = (*speckle_fg_size, 10)
        speckle_bg_size = (*speckle_bg_size, 10)
    for imgfile in imgfiles:
        if random.random() > 0.75: continue
        # Using PIL to read image as grayscale
        img = np.asarray(Image.open(imgfile).convert('L'))/255
        img_degraded = img

        # Filtering section
        # 1 Randomly rotate image
        if 'rotate' in degrade_filters:
            img_degraded = ndi.rotate(img_degraded, 0)

        # 2 Transform image in various ways
        if 'transform' in degrade_filters:
            img_degraded = ocrodeg.transform_image(
                img_degraded,
                angle=random_func(*rotate),
                aniso=random_func(*ansio),
                scale=random_func(*scale),
                translation=(random_func(*translation), random_func(*translation))
            )

        # 3 Add distortion and degradation
        if 'noise' in degrade_filters:
            gaussian_deltas = ocrodeg.bounded_gaussian_noise(img_degraded.shape, 0, random_func(*gaussian_noise))
            img_degraded = ocrodeg.distort_with_noise(img_degraded, gaussian_deltas)
            flat_noise_deltas = ocrodeg.noise_distort1d(img_degraded.shape, magnitude=flat_noise)
            img_degraded = ocrodeg.distort_with_noise(img_degraded, flat_noise_deltas)

        # 4 Add blur
        if 'blur' in degrade_filters:
            img_degraded = ndi.gaussian_filter(img_degraded, random_func(*gaussian_filter))

        # 5 Binarize and blur
        if 'binary_blur' in degrade_filters:
            img_degraded = ocrodeg.binary_blur(img_degraded, random_func(0.1, 1.5), noise=random_func(*binary_blur))

        # 5 Add random blotches
        if speckle is not None:
            max_value = np.mean(img_degraded[img_degraded > 0.5])
            if 'fg' in speckle:
                fg = ocrodeg.random_blobs(img_degraded.shape, random_func(*speckle_fg_density),
                                          random_func(*speckle_fg_size), speckle_fg_roughness)
            else:
                fg = np.zeros(img_degraded.shape, 'i')
            if 'bg' in speckle:
                bg = ocrodeg.random_blobs(img_degraded.shape, random_func(*speckle_bg_density),
                                          random_func(*speckle_bg_size), speckle_bg_roughness)*max_value
            else:
                bg = np.zeros(img_degraded.shape, 'i')
            blurred = ndi.gaussian_filter(ocrodeg.minimum(ocrodeg.maximum(img_degraded, fg), max_value - bg), 1.0)
            # Add speckles
            if not printlike:
                img_degraded = blurred
            elif printlike == 'multiscale':
                paper = ocrodeg.make_multiscale_noise_uniform(img_degraded.shape, limits=(0.5, 1.0))
                ink = ocrodeg.make_multiscale_noise_uniform(img_degraded.shape, limits=(0.0, 0.5))
                img_degraded = (1 - blurred) * ink + blurred * paper
            elif printlike == 'fibrous':
                paper = ocrodeg.make_multiscale_noise(img_degraded.shape, [1.0, 5.0, 10.0, 50.0], weights=[1.0, 0.3, 0.5, 0.3],
                                              limits=(0.7, 1.0))
                paper -= ocrodeg.make_fibrous_image(img_degraded.shape, 300, 500, 0.01, limits=(0.0, 0.25), blur=0.5)
                ink = ocrodeg.make_multiscale_noise(img_degraded.shape, [1.0, 5.0, 10.0, 50.0], limits=(0.0, 0.5))
                img_degraded = (1 - blurred) * ink + blurred * paper

        # 6 Apply fibrous_patch and/or speckle
        elif printlike:
            if 'binary_blur' not in degrade_filters:
                img_degraded = 1.0 * (img_degraded > 0.5)
            if printlike == 'fibrous':
                img_degraded = ocrodeg.printlike_fibrous(img_degraded)
            elif printlike == 'multiscale':
                img_degraded = ocrodeg.printlike_multiscale(img_degraded)

        # Save edited image
        if not output_dir:
            new_filename = imgfile.parent.joinpath(imgfile.with_suffix('').name+'_edit.png')
        else:
            new_filename = output_dir.joinpath(imgfile.with_suffix('').name+'_edit.png')
            new_filename.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_degraded*255).convert('L').save(new_filename)
        print(f'Degraded "{new_filename.name}" >>> Saved to {new_filename.resolve()}.')


if __name__ == '__main__':
    degrade()