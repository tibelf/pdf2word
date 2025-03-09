# -*- coding: utf-8 -*-

'''Page object parsed with PDF raw dict.

In addition to base structure described in :py:class:`~pdf2docx.page.RawPage`, 
some new features, e.g. sections, table block, are also included. 
Page elements structure:

* :py:class:`~pdf2docx.page.Page` >> :py:class:`~pdf2docx.layout.Section` >> :py:class:`~pdf2docx.layout.Column`  
    * :py:class:`~pdf2docx.layout.Blocks`
        * :py:class:`~pdf2docx.text.TextBlock` >> 
          :py:class:`~pdf2docx.text.Line` >> 
          :py:class:`~pdf2docx.text.TextSpan` / :py:class:`~pdf2docx.image.ImageSpan` >>
          :py:class:`~pdf2docx.text.Char`
        * :py:class:`~pdf2docx.table.TableBlock` >>
          :py:class:`~pdf2docx.table.Row` >> 
          :py:class:`~pdf2docx.table.Cell`
            * :py:class:`~pdf2docx.layout.Blocks`
            * :py:class:`~pdf2docx.shape.Shapes`
    * :py:class:`~pdf2docx.shape.Shapes`
        * :py:class:`~pdf2docx.shape.Shape.Stroke`
        * :py:class:`~pdf2docx.shape.Shape.Fill`
        * :py:class:`~pdf2docx.shape.Shape.Hyperlink`

::

    {
        "id": 0, # page index
        "width" : w,
        "height": h,
        "margin": [left, right, top, bottom],
        "sections": [{
            ... # section properties
        }, ...],
        "floats": [{
            ... # floating picture
        }, ...]
    }

'''

from docx.shared import Pt
from docx.enum.section import WD_SECTION
from ..common.Collection import BaseCollection
from ..common.share import debug_plot
from .BasePage import BasePage
from ..layout.Sections import Sections
from ..image.ImageBlock import ImageBlock
from docx.document import Document
import logging


class Page(BasePage):
    '''Object representing the whole page, e.g. margins, sections.'''

    def __init__(self, id:int=-1, 
                        skip_parsing:bool=True,
                        width:float=0.0,
                        height:float=0.0,
                        header:str=None, 
                        footer:str=None, 
                        margin:tuple=None, 
                        sections:Sections=None,
                        float_images:BaseCollection=None):
        '''Initialize page layout.

        Args:
            id (int, optional): Page index. Defaults to -1.
            skip_parsing (bool, optional): Don't parse page if True. Defaults to True.
            width (float, optional): Page width. Defaults to 0.0.
            height (float, optional): Page height. Defaults to 0.0.
            header (str, optional): Page header. Defaults to None.
            footer (str, optional): Page footer. Defaults to None.
            margin (tuple, optional): Page margin. Defaults to None.
            sections (Sections, optional): Page contents. Defaults to None.
            float_images (BaseCollection, optional): Float images in th is page. Defaults to None.
        ''' 
        # page index
        self.id = id
        self.skip_parsing = skip_parsing

        # page size and margin
        super().__init__(width=width, height=height, margin=margin)

        # flow structure: 
        # Section -> Column -> Blocks -> TextBlock/TableBlock
        # TableBlock -> Row -> Cell -> Blocks
        self.sections = sections or Sections(parent=self)

        # page header, footer
        self.header = header or ''
        self.footer = footer or ''
        
        # floating images are separate node under page
        self.float_images = float_images or BaseCollection()

        self._finalized = False


    @property
    def finalized(self): return self._finalized   


    def store(self):
        '''Store parsed layout in dict format.'''
        res = {
            'id'      : self.id,
            'width'   : self.width,
            'height'  : self.height,
            'margin'  : self.margin,
            'sections': self.sections.store(),
            'header'  : self.header,
            'footer'  : self.footer,
            'floats'  : self.float_images.store()
        }
        return res


    def restore(self, data:dict):
        '''Restore Layout from parsed results.'''
        # page id
        self.id = data.get('id', -1)

        # page width/height
        self.width = data.get('width', 0.0)
        self.height = data.get('height', 0.0)
        self.margin = data.get('margin', (0,) * 4)
        
        # parsed layout
        self.sections.restore(data.get('sections', []))
        self.header = data.get('header', '')
        self.footer = data.get('footer', '')

        # float images
        self._restore_float_images(data.get('floats', []))

        # Suppose layout is finalized when restored; otherwise, set False explicitly
        # out of this method.
        self._finalized = True

        return self


    @debug_plot('Final Layout')
    def parse(self, **settings):
        '''Parse page layout.'''
        self.sections.parse(**settings)
        self._finalized = True
        return self.sections # for debug plot


    def extract_tables(self, **settings):
        '''Extract content from tables (top layout only).
        
        .. note::
            Before running this method, the page layout must be either parsed from source 
            page or restored from parsed data.
        '''
        # table blocks
        collections = []        
        for section in self.sections:
            for column in section:
                if settings['extract_stream_table']:
                    collections.extend(column.blocks.table_blocks)
                else:
                    collections.extend(column.blocks.lattice_table_blocks)
        
        # check table
        tables = [] # type: list[ list[list[str]] ]
        for table_block in collections:
            tables.append(table_block.text)

        return tables


    def make_docx(self, doc:Document=None, **kwargs):
        """Create docx with layout extracted from PDF page.

        Args:
            doc (Document, optional): ``python-docx`` Document instance. 
                A new one created if None. Defaults to None.

        Returns:
            Document: ``python-docx`` Document instance.
        """
        # new document if not provided
        if not doc:
            doc = Document()
            # page section
            section = doc.sections[0]
            section.page_width = Pt(self.width)
            section.page_height = Pt(self.height)
            section.left_margin = Pt(self.margin[0])
            section.right_margin = Pt(self.margin[1])
            section.top_margin = Pt(self.margin[2])
            section.bottom_margin = Pt(self.margin[3])
        
        # process formula regions if provided
        formula_regions = []
        if kwargs.get('process_formulas') and 'formula_regions_by_page' in kwargs:
            formula_regions = kwargs.get('formula_regions_by_page', {}).get(self.id, [])
        
        # process sections
        for section in self.sections:
            # try:
            section.make_docx(doc, **kwargs)
            # except Exception as e:
            #     if kwargs.get('debug', False): raise e

        # insert formulas if available
        if formula_regions:
            self._insert_formulas(doc, formula_regions)
            
        return doc
    
    def _insert_formulas(self, doc, formula_regions):
        """Insert mathematical formulas into the document.
        
        Args:
            doc (Document): python-docx Document instance.
            formula_regions (list): List of formula regions with mathml content.
        """
        try:
            from docx.oxml.ns import qn
            from docx.oxml import OxmlElement
            import re
            
            # Sort formula regions by vertical position (top to bottom)
            formula_regions.sort(key=lambda x: x['pdf_bbox'][1])
            
            # Process each formula
            for region in formula_regions:
                if not region.get('mathml'):
                    continue
                    
                # Add a paragraph for the formula
                p = doc.add_paragraph()
                
                # Try to insert the MathML as an OfficeMath object
                mathml = region.get('mathml')
                
                # Try to insert using the raw XML
                r = p.add_run()
                
                # Create an office math element
                omath = OxmlElement('m:oMath')
                
                # Try to parse and include the MathML
                # This is a simplified approach - in a real implementation
                # you would convert MathML to Office Math XML (OMML)
                math_text = re.sub(r'<[^>]+>', '', mathml)
                omath.text = math_text
                
                # Create an office math paragraph
                omathpara = OxmlElement('m:oMathPara')
                omathpara.append(omath)
                
                # Add a math region to the run's XML
                r._r.append(omathpara)
                
                # Add a caption with the LaTeX representation
                if region.get('latex'):
                    caption = doc.add_paragraph(f"LaTeX: {region.get('latex')}")
                    caption.style = 'Caption'
        except Exception as e:
            logging.error(f"Error inserting formulas: {str(e)}")
            # Continue without inserting the formula


    def _restore_float_images(self, raws:list):
        '''Restore float images.'''
        self.float_images.reset()
        for raw in raws:
            image = ImageBlock(raw)
            image.set_float_image_block()
            self.float_images.append(image)
