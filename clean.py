import re

def caption(caption):
	"""
	Will some replace characters with no space, and others with a space. Will remove extra spaces and make lower.
	"""

	def replaceText(text, chars, replace):
		for c in chars:
			text = text.replace(c, replace)
		return text
	return re.sub(' +',' ', replaceText(replaceText(caption, ":`~;()[].!?-\n\"\'", ""), "\n@#$%^&*0123456789\\/,", " ")).lower().strip()