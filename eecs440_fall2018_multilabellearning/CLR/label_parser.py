import xml.sax

file = "./Datasets/scene/scene.xml"

class LabelHandler(xml.sax.ContentHandler):
	def __init__(self):
		self.labels=[]
		
	def startElement(self, tag, attributes):
		if tag == "label":
			self.labels.append(attributes["name"])
			
def LabelParser(file):
	parser = xml.sax.make_parser()
	parser.setFeature(xml.sax.handler.feature_namespaces, 0)
	Handler = LabelHandler()
	parser.setContentHandler(Handler)
	parser.parse(file)
	return Handler.labels
			
if __name__ == "__main__":
	parser = xml.sax.make_parser()
	parser.setFeature(xml.sax.handler.feature_namespaces, 0)
	Handler = LabelHandler()
	parser.setContentHandler(Handler)
	parser.parse(file)
	print(Handler.labels)