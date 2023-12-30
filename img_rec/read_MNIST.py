import numpy as np
import struct


def loadImageSet( model ):
	imgFile = None
	if model == 'train':
		imgFile = open('./mnist/train-images.idx3-ubyte', 'rb')
	elif model == 'test':
		imgFile = open('./mnist/t10k-images.idx3-ubyte', 'rb')
	else:
		print('Wrong Input. train or test')
	buf = imgFile.read()

	magic, imgNum, rowNum, ColNum = struct.unpack_from('>IIII', buf, 0)
	offset = struct.calcsize('>IIII')
	#In trainSet, [60000]*[28*28]
	bits = imgNum * rowNum * ColNum
	# print(bits)
	bitString = '>' + str(bits) + 'B'
	imgs = struct.unpack_from(bitString, buf, offset)
	imgFile.close()

	imgs = np.reshape(imgs, [imgNum, rowNum*ColNum])

	return imgs


def loadLabelSet( model ):
	labelFile = None
	if model == 'train':
		labelFile = open('./mnist/train-labels.idx1-ubyte', 'rb')
	elif model == 'test':
		labelFile = open('./mnist/t10k-labels.idx1-ubyte', 'rb')
	else:
		print('Worng Input. train or test')
	buf = labelFile.read()

	magic, imgNum = struct.unpack_from('>II', buf, 0)
	offset = struct.calcsize('>II')

	bitString = '>' + str(imgNum) + 'B'
	labels = struct.unpack_from(bitString, buf, offset)
	labelFile.close()

	labels = np.reshape(labels, [imgNum, 1])

	return labels

