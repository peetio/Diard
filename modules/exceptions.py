#   Custom Exceptions
class DocumentFileFormatError(Exception):
    """Raised when given document file format not supported"""

    def __init__(self, name, file_format):

        if len(file_format) > 1:
            self.message = 'Document \'{}\' has an unsupported file format: \'.{}\''.format(name, file_format)
        else:
            self.message = 'Document \'{}\' has no file format extension'.format(name)

    def __str__(self):
        return self.message



class UnsetAttributeError(Exception):
    """Raised when a required attribute is not set (is None)"""

    def __init__(self, method, attributes):
        self.message = 'Cannot call \'{}\' method before following attributes are set: {}'.format(method, attributes)

    def __str__(self):
        return self.message



class PageNumberError(Exception):
    """Raised when user tries to access non-existing document pages"""

    def __init__(self, page, pages):
        self.message = 'Cannot access item at index {}, document only consists out of {} pages. Please note that zero-based numbering is used.'.format(page, pages)

    def __str__(self):
        return self.message



class InputJsonStructureError(Exception):
    """Raised when content structure of JSON file isn't supported"""

    def __init__(self, filename):
        self.message = 'Content structure of file \'' + filename + '\' is not supported.'

    def __str__(self):
        return self.message

