'''Entry for ``pdf2word`` command line.'''
import logging
import sys
import fire
from .converter import Converter


def convert(pdf_file:str,
            docx_file:str=None,
            password:str=None,
            start:int=0,
            end:int=None,
            pages:list=None,
            multi_processing:bool=False,
            **kwargs):
    '''Convert pdf file to docx file.

    Args:
        pdf_file (str) : PDF filename to read from.
        docx_file (str, optional): docx filename to write to. Defaults to None.
        password (str): Password for encrypted pdf. Default to None if not encrypted.
        start (int, optional): First page to process. Defaults to 0.
        end (int, optional): Last page to process. Defaults to None.
        pages (list, optional): Range of pages, e.g. --pages=1,3,5. Defaults to None.
        multi_processing (bool, optional): Whether to use multi-processing. Defaults to False.
        kwargs (dict) : Configuration parameters.
    '''
    # Ensure kwargs contains required defaults
    kwargs.setdefault('raw_exceptions', False)
    kwargs.setdefault('zero_based_index', True)
    
    # index starts from zero or one
    if isinstance(pages, int): pages = [pages] # in case --pages=1
    if not kwargs.get('zero_based_index', True):
        start = max(start-1, 0)
        if end: end -= 1
        if pages: pages = [i-1 for i in pages]

    cv = Converter(pdf_file, password)
    try:
        # Include multi_processing in kwargs instead of as a separate argument
        kwargs['multi_processing'] = multi_processing
        cv.convert(docx_file, start, end, pages, **kwargs)
    except Exception as e:
        logging.error(e)
        if kwargs.get('raw_exceptions', False):
            raise
    finally:
        cv.close()


def parse(pdf_file:str, 
          start:int=0, 
          end:int=None, 
          pages:list=None, 
          **kwargs):
    '''Parse and extract contents from pdf file.

    Args:
        pdf_file (str) : PDF filename to read from.
        start (int, optional): First page to process. Defaults to 0.
        end (int, optional): Last page to process. Defaults to None.
        pages (list, optional): Range of pages, e.g. --pages=1,3,5. Defaults to None.
        kwargs (dict) : Configuration parameters.
    '''
    # Ensure kwargs contains required defaults
    kwargs.setdefault('raw_exceptions', False)
    kwargs.setdefault('zero_based_index', True)
    
    # index starts from zero or one
    if isinstance(pages, int): pages = [pages] # in case --pages=1
    if not kwargs.get('zero_based_index', True):
        start = max(start-1, 0)
        if end: end -= 1
        if pages: pages = [i-1 for i in pages]

    cv = Converter(pdf_file)
    try:
        # Create empty pages
        return cv._pages
    except Exception as e:
        logging.error(e)
        if kwargs.get('raw_exceptions', False):
            raise
    finally:
        cv.close()


class CLI:
    '''Command line interface for pdf2word.'''

    def convert(self, pdf_file:str,
                docx_file:str=None,
                password:str=None,
                start:int=0,
                end:int=None,
                pages:list=None,
                multi_processing:bool=False,
                **kwargs):
        '''Convert pdf file to docx file.'''
        # Pass multi_processing as a positional argument since the convert function expects it
        convert(pdf_file, docx_file, password, start, end, pages, multi_processing, **kwargs)

    def table(self, pdf_file:str,
              password:str=None,
              start:int=0,
              end:int=None,
              pages:list=None,
              **kwargs):
        '''Extract tables from pdf file.'''
        cv = Converter(pdf_file, password)
        try:
            tables = cv.extract_tables(start, end, pages, **kwargs)
            return tables
        except Exception as e:
            logging.error(e)
            return []
        finally:
            cv.close()


def main():
    '''Entry point for pdf2word command.'''
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format="[%(levelname)s] %(message)s")
    
    # Handle command arguments
    fire.Fire(CLI)


if __name__ == '__main__':
    main()
