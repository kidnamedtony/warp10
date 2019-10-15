import logging

# Creating a custom logger object:
logger = logging.getLogger("webscraping_helpers")

# Handlers for the logger oject:
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler("Progress.log", "w")
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.DEBUG)

# Formatter to set time/date formate for the handlers to output:
c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to logger object:
logger.addHandler(c_handler)
logger.addHandler(f_handler)
