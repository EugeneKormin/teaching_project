import configparser


config = configparser.ConfigParser()
config.sections()


config.read('../teaching_project/config.ini')
PATH_TO_IMAGES: str = config['Paths']['to_images']
PATH_TO_TEMPLATES: str = config['Paths']['to_templates']
PATH_TO_RESULTS: str = config['Paths']['to_results']
PATH_TO_HAARCASCADE: str = config['Paths']['to_haarcascade']
