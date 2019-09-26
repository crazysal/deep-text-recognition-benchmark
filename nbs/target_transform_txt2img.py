import os
from PIL import Image, ImageDraw, ImageFont

def get_gt_vocab():
	return [None, None]

def render_img(v, save_dir=None, ret=True):
	# make sure you have the fonts locally in a fonts/ directory
	georgia_bold = 'fonts/georgia_bold.ttf'
	georgia_bold_italic = 'fonts/georgia_bold_italic.ttf'
	open_sans_regular = './fonts/OpenSans-Regular.ttf'

	# W, H = (1280, 720) # image size
	W, H = (400, 160) # image size
	# txt = 'Hello Petar this is my test image' # text to render
	txt = v
	background = (255,255,255) # white
	fontsize = 45
	font = ImageFont.truetype(open_sans_regular, fontsize)

	image = Image.new('RGBA', (W, H), background)
	draw = ImageDraw.Draw(image)

	# w, h = draw.textsize(txt) # not that accurate in getting font size
	w, h = font.getsize(txt)
	# print("font get size", w,h)
	
	if w > W :
		w_ = w 
	else :
		w_ = (W-w)/2
	if h > H :
		h_ = h 
	else :
		h_ = (H-h)/2

	draw.text((w_,h_), txt, fill='black', font=font)

	# save_location = os.getcwd()
	save_location = save_dir

	# img_resized.save(save_location + '/sample.jpg')
	if ret :
		return image
	else : 
		image.save(save_location + '/sample.png')


def main():
	vocab = get_gt_vocab()
	save_dir = "/home/sahmed9/Documents/reps/deep-text-recognition-benchmark/demo_image/" 
	for v in vocab : 
		render_img(v, save_dir)


if __name__ == '__main__':
	main()