# import libraries
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import numpy
import sys
import zipfile
import pathlib
import os


def blur_face(data):
	""" Detect and blur faces found in image

	Faces are downsized to small rect (max 10,10) and resized back to make them unrecognizable.
	Faces smaller than 10x10 are assumed to be unrecognizable without blur.

	data -- image file bytes
	return -- cv2 image
	"""
	mat=numpy.asarray(bytearray(data), dtype=numpy.uint8)
	im=cv2.imdecode(mat, cv2.IMREAD_COLOR)

	# determines information loss when pixelating/blurring face
	scale=0.1

	faces, confidences = cv.detect_face(im)
	# blur all faces
	for face in faces:
		x0,y0,x1,y1=face
		w,h=x1-x0,y1-y0

		# determine blur strength
		# calc scale to preserve aspect ratio
		scale=min(10/w,10/h)
		face_downsize=(max(int(scale*w),1),max(int(scale*h),1))

		# blur/pixelate face
		# tinker with interpolation parameters for various effects (ex. for pixelize use INTER_AREA, INTER_NEAREST)
		im_face=im[y0:y1,x0:x1]
		im_face=cv2.resize(im_face, face_downsize, interpolation = cv2.INTER_AREA)
		im_face=cv2.resize(im_face, (w,h), interpolation = cv2.INTER_LINEAR)
		im[y0:y1,x0:x1]=im_face

	return im



def blur_faces_from_zip(ipath_zip, opath_dir):
	""" Iterate images found in zip file, blur faces and write to output dir.
	ipath_zip -- zip containing input images
	opath_dir -- where to write output images
	effect -- output images are written to opath_dir directory
	"""
	if not os.path.exists("output"):
		print("I creating output directory")
		os.makedirs(opath_dir)

	with zipfile.ZipFile(ipath_zip, "r") as f:
		for name in f.namelist():
			path=pathlib.Path(name)
			if not name.endswith("/"):
				data=f.read(name)
				im=blur_face(data)
				opath=opath_dir/path
				print("I {}".format(opath))
				cv2.imwrite(str(opath),im)
			else:
				if not os.path.exists(opath_dir/path):
					os.makedirs(opath_dir/path)


def main():
	if len(sys.argv)!=2:
		print("Usage: python3 main.py <faces.zip>")
		sys.exit(0)

	ipath_zip=sys.argv[1]
	blur_faces_from_zip(ipath_zip, "output")


if __name__=="__main__":
	main()
