import os
import re
import pandas as pd
from tqdm import tqdm
from Bio import Entrez
import xml.etree.ElementTree as ET

class ProcessPubmed:
    def __init__(self, outdir, predDir='./predicion_input'):
        Entrez.email = 'ycfrenchgirl2@gmail.com'
        Entrez.api_key="4543094c8a41e6aecf9a1431bff42cfac209"
        
        self.out = outdir
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
        
        self.pred = predDir
        if( not os.path.isdir( self.pred ) ) :
            os.makedirs( self.pred )
    
    def _parse_paper(self, id):
        fetch = Entrez.efetch(db='pubmed', resetmode='text',id = str(id), rettype='abstract')
        ftmp = os.path.join( self.out, f'{id}_tempFile.xml' )
        with open( ftmp, 'wb') as f:
            f.write(fetch.read())
        
        lines = []
        txt = open( ftmp, 'r').read()
        if( txt.find('Abstract>') != -1 ):
            if( txt.find('PublicationTypeList>') != -1 ):
                ptst = '<PublicationTypeList>'+ txt.split('<PublicationTypeList>')[1].split('</PublicationTypeList>')[0] +'</PublicationTypeList>'
            ptypes = []
            tree = ET.ElementTree( ET.fromstring( ptst ) )
            root = tree.getroot()
            for at in root.findall('PublicationType'):
                ptypes.append(at.text)
            ptypes = '|'.join(ptypes)
            
            abst = '<Abstract>'+ txt.split('<Abstract>')[1].split('</Abstract>')[0] +'</Abstract>'
            tree = ET.ElementTree( ET.fromstring( abst ) )
            root = tree.getroot()
            for at in root.findall('AbstractText'):
                lab = ''
                if( 'Label' in at.attrib ):
                    lab = at.attrib['Label']
                    
                txt = at.text
                chex = set( list(re.findall('&#x([a-z0-9]+);', txt)) )
                for it in chex:
                    try:
                        txt = txt.replace( f"&#x{it};", bytes.fromhex(str(it)).decode('utf-8') )
                    except:
                        pass
                lines.append( [ id, lab, ptypes, txt ] )
        os.remove(ftmp)
        
        return lines
        
    def get_abstracts(self):
        omap = os.path.join( self.out, 'abstract_info_pubmed.tsv' )
        fm = open( omap, 'w' )
        fm.write("pmid\tlabel\tpublication_type\ttext\n")
        fm.close()
    
        inmap = os.path.join( self.out, 'mapping_pubmed.tsv' )
        df = pd.read_csv( inmap, sep='\t')
        ids = set(df['pmid'].values)
        for id in tqdm(ids):
            try:
                lines = self._parse_paper( str(id) )
                if( len(lines) > 0 ):
                    lines = list( map( lambda x: ("\t".join(x)), lines ))
                    with open( omap, 'a' ) as fm:
                        fm.write( "\n".join(lines)+"\n" )
            except:
                print( 'error', id)
    
    def generate_txt_inputs(self):
        inmap = os.path.join( self.out, 'mapping_pubmed.tsv' )
        df = pd.read_csv( inmap, sep='\t')
        ids = set( df.pmid.unique() )
        df = df[ df.text.str.contains('randomised') ]
        rctids = set( df.pmid.unique() )
        print( len(rctids), '/', len(ids), 'are randomized' )
        txts = {}
        for i in df.index:
            pmid = df.iloc[i,'pmid']
            lab = df.iloc[i, 'label']
            fname = f"{pmid}_{lab.lower()}.txt"
            if( not fname in txts):
                txts[fname] = []
            text = df.iloc[i, 'text']
            txts[fname].append(text)
        
        for fname in txts:
            text = ' '.join( txts[fname] )
            opath = os.path.join( self.pred, fname )
            f = open(opath, 'w')
            f.write(text)
            f.close()
        
    def run(self):
        #self.get_abstracts()
        self.generate_txt_inputs()
        
if( __name__ == "__main__" ):
    odir = './out'
    i = ProcessPubmed( odir )
    i.run()
