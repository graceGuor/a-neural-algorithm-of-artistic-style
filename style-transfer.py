import numpy as np
from tensorflow import set_random_seed
from neural_stylization.transfer_style import Stylizer
from neural_stylization.optimizers import GradientDescent, L_BFGS, Adam
from neural_stylization.util.build_callback import build_callback
from neural_stylization.util.img_util import load_image

np.random.seed(1)
set_random_seed(1)
# CONTENT = 'img/content/000000000885.jpg'
# CONTENT = '../../data/style_transfer/cases/content/cat.jpg'
CONTENT = 'img/content/cat.jpg'
# CONTENT = 'img/content/avatar.jpg'
# CONTENT = 'img/content/tubingen.jpg'
load_image(CONTENT)
DIMS = None

sty = Stylizer(content_weight=1e-5, style_weight=1)
seated_nudes = sty(
    content_path=CONTENT,
    # style_path='../../data/style_transfer/cases/styles/tiger.png',
    # style_path='img/styles/colorful-girl.png',
    # style_path='img/styles/tiger.png',
    # style_path='img/styles/tiger.png',
    # style_path='img/styles/the-starry-night.jpg',
    style_path='img/styles/seated-nude.jpg',
    optimize=L_BFGS(),
    iterations=30,
    image_size=DIMS,
    save_freq=10,
    initialization_strat='content',
    # save_path='img/res/colorful-girl.png',
    # save_path='img/res/tiger.png',
    # save_path='img/res/the-starry-night.png',
    save_path='img/res/seated-nude.png',
    # callback=build_callback('build/transfer/colorful-girl')
    # callback=build_callback('build/transfer/the-starry-night')
    callback=build_callback('build/transfer/seated-nude')
)
# seated_nudes.save('img/res/000000000885-the-starry-night.png')
# seated_nudes.save('img/res/avatar-the-starry-night.png')
# seated_nudes.save('img/res/cat-tiger.png')
seated_nudes.save('img/res/cat-seated-nude.png')

print("initialization_strat = 'noise',Finished!")
# print("initialization_strat = 'content',Finished!")
