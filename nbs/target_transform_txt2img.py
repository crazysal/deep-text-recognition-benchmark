import os
from PIL import Image, ImageDraw, ImageFont

def get_gt_vocab():
	return [None, None]

def render_img(v, save_dir):
	# make sure you have the fonts locally in a fonts/ directory
	georgia_bold = 'fonts/georgia_bold.ttf'
	georgia_bold_italic = 'fonts/georgia_bold_italic.ttf'
	open_sans_regular = './fonts/OpenSans-Regular.ttf'

	# W, H = (1280, 720) # image size
	W, H = (200, 60) # image size
	txt = 'Hello Petar this is my test image' # text to render
	background = (255,255,255) # white
	fontsize = 35
	font = ImageFont.truetype(open_sans_regular, fontsize)

	image = Image.new('RGBA', (W, H), background)
	draw = ImageDraw.Draw(image)

	# w, h = draw.textsize(txt) # not that accurate in getting font size
	w, h = font.getsize(txt)
	print("font get size", w,h)
	draw.text(((W-w)/2,(H-h)/2), txt, fill='black', font=font)
	# draw.text((10, 0), txt, (0,0,0), font=font)
	# img_resized = image.resize((188,45), Image.ANTIALIAS)

	save_location = os.getcwd()

	# img_resized.save(save_location + '/sample.jpg')
	image.save(save_location + '/sample.png')

def main():
	vocab = get_gt_vocab()
	save_dir = "/home/sahmed9/Documents/reps/deep-text-recognition-benchmark/demo_image/" 
	for v in vocab : 
		render_img(v, save_dir)


if __name__ == '__main__':
	main()