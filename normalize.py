import statistics

pixels_normalized = []

def normalize(pixels):

	for i in range(len(pixels)):
		image_mean = statistics.mean(pixels[i])
		standard_deviation = statistics.stdev(pixels[i])
		temp = []
		for j in range(len(pixels[i])):
			normalized_pixel = (pixels[i][j] - image_mean) / standard_deviation
			temp.append(normalized_pixel)
		pixels_normalized.append(temp)

	return pixels_normalized