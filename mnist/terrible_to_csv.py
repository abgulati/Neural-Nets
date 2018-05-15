#source: https://pjreddie.com/projects/mnist-in-csv/
#referenced and understood from the above link
#comments added here to clarify the code

def convert(img_file, label_file, output_file, no_of_samples):
	f = open(img_file, "rb")	# 'b' indicates reading a binary file, not a text file
	o = open(output_file, "w")
	l = open(label_file, "rb")	# 'rb' is "read a binary file"

	f.read(16)		#skipping the first 16 Bytes in our images.idx file (see below)
	l.read(8)		#skipping the first 8 Bytes in our labesl.idx file (see below)
	images = []

	#for each image, we append it's label first
	for i in range(no_of_samples):
		image = [ord(l.read(1))]		#ord() gives either the ASCII value for a char or the value of the byte for a 8-bit string
		for j in range(28*28):				#ord() is an abbriviation for ordinal numbers, which means countable decimal numbers
			image.append(ord(f.read(1)))	
		images.append(image)

	for image in images:
		o.write(",".join(str(pix) for pix in image)+"\n")		#writing into a Comma Separated Value (CSV) file

	f.close()
	o.close()
	l.close()

convert("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
		"mnist_train.csv", 60000)
convert("t10k-images.idx3-ubyte","t10k-labels.idx1-ubyte",
		"mnist_test.csv", 10000)


#For each of the 60k images, we're reading 784 Bytes as seen above (f.read(1) 784 times/image). This is because for a 28x28
#image, that's 784 pixels and each pixel is represented by a single byte. Those 784 Bytes are read one at a time. The first
#16 Bytes are skipped(see below), then the read commences for the remaining Bytes, one Byte at a time. For 60000 images, 
#784 Bytes per image is 60000*784 = 47040000 Bytes in total. In our IDX file, we have 16 bytes per line, so 
#that's 470470000/16 = 2940000 lines, and we have exactly 2940001 full lines in our image IDX file and we are skipping
#the first 16 Bytes, or one line.

#Similarly, for the labels, we're skipping the first 8 Bytes. For 60000 images, each label is given by a single Byte, that's 
#60000 Bytes of label data for 60000 images. We have 16 Bytes per line, so thats 60000/16 = 3750 lines. Our IDX file has 
#exactly 3750 lines with 16 Bytes each, and a 3751th line with 8 Bytes, but 8 Bytes are skipped at the start.

#Why are the first 16 Bytes skipped in our image files and the first 8 bytes skipped in our label files? This is because in
#the image file, pixel data starts from the 17th Byte, and in the label file, label data starts from the 9th Byte. In the
#image file, the first 0-15 Bytes are used by the magic number, the number of images, the number of rows, and the number of
#columns, data that takes 4Bytes each. For the label files, the magic number and number of items take up the first 8 Bytes.
#The remaining bytes are then unsigned pixel/unsigned label data in the image and label IDX files respectively.