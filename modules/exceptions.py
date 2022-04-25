#   Custom Exceptions
class DocumentFileFormatError(Exception):
    """Raised when given document file format not supported"""

    def __init__(self, filename):
        self.filename = filename

        split_filename = filename.split('.')
        if len(split_filename) > 1:
            self.extension = split_filename[-1]
            self.message = 'Document \'{}\' has an unsupported file format: \'.{}\''.format(self.filename, self.extension)
        else:
            self.message = 'Document \'{self.filename}\' has no file format extension'

    def __str__(self):
        return self.message



class UnsetAttributeError(Exception):
    """Raised when a required attribute is not set (is None)"""

    def __init__(self, method, attributes):
        self.message = 'Cannot call \'{}\' method before following attributes are set: {}'.format(method, attributes)

    def __str__(self):
        return self.message
