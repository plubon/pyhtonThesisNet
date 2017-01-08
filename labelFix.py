def main():
	labelsfile = open("labels", 'r')
	fixefLabelsFile =open("fixedLabels", "w")
	for line in labelsfile:
	    tokens = line.split(";")
	    fixedLine = str(int(tokens[0])-1)+";"+tokens[1]
	    fixefLabelsFile.write(fixedLine)
	labelsfile.close()
	fixefLabelsFile.close()

if __name__ == "__main__":
    main()