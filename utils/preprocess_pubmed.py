import os
import re
import logging
import pandas as pd
from tqdm import tqdm
from Bio import Entrez
import xml.etree.ElementTree as ET

class ProcessPubmed:
    def __init__(self, outdir, predDir='./prediction_input'):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='log_preprocess_pubmed.log', level=logging.INFO)
        
        Entrez.email = 'ycfrenchgirl2@gmail.com'
        Entrez.api_key="4543094c8a41e6aecf9a1431bff42cfac209"
        
        self.out = outdir
        if( not os.path.isdir( self.out ) ) :
            os.makedirs( self.out )
        
        self.pred = predDir
        if( not os.path.isdir( self.pred ) ) :
            os.makedirs( self.pred )
    
    def _parse_paper(self, id):
        ftmp = os.path.join( self.out, f'{id}_tempFile.xml' )
        if( not os.path.isfile(ftmp) ):
            fetch = Entrez.efetch(db='pubmed', resetmode='text',id = str(id), rettype='abstract')
            with open( ftmp, 'wb') as f:
                f.write(fetch.read())
        
        lines = []
        txt = open( ftmp, 'r').read()
        if( txt.find('<Abstract>') != -1 ):
            idst = ''
            id_ = str(id)
            if( txt.find('</PMID>') != -1 ):
                id_ = txt.split('</PMID>')[0].split('>')[-1]
        
            ptst = ''
            ptypes = []
            if( txt.find('PublicationTypeList>') != -1 ):
                ptst = '<PublicationTypeList>'+ txt.split('<PublicationTypeList>')[1].split('</PublicationTypeList>')[0] +'</PublicationTypeList>'
                tree = ET.ElementTree( ET.fromstring( ptst ) )
                root = tree.getroot()
                for at in root.findall('PublicationType'):
                    ptypes.append(at.text)
            ptypes = '|'.join(ptypes)
            
            print(id)
            abst = '<Abstract>'+ txt.split('<Abstract>')[1].split('</Abstract>')[0] +'</Abstract>'
            tree = ET.ElementTree( ET.fromstring( abst ) )
            root = tree.getroot()
            for at in root.findall('AbstractText'):
                lab = ''
                if( 'Label' in at.attrib ):
                    lab = at.attrib['Label']
                    
                txt = str(at.text)
                #try:
                chex = set( list(re.findall('&#x([a-z0-9]+);', txt)) )
                for it in chex:
                    try:
                        txt = txt.replace( f"&#x{it};", bytes.fromhex(str(it)).decode('utf-8') )
                    except:
                        pass
                #except:
                #    pass
                    
                lines.append( [ id, lab, ptypes, txt ] )
        os.remove(ftmp)
        
        return lines
    
    def _parse_paper_group(self, fname):
        txt = open(fname, 'r').read()
        tree = ET.ElementTree( ET.fromstring( txt ) )
        root = tree.getroot()
        lines = []
        for pm in root.findall('PubmedArticle'):
            md = pm.findall('MedlineCitation')[0]
            pmid = md.findall('PMID')[0].text
            print('----', pmid)
            article = md.findall('Article')[0]
            ptypes = []
            nodesp = article.findall('PublicationTypeList')
            if( len(nodesp) > 0 ):
                pubtype = nodesp[0]
                for at in pubtype.findall('PublicationType'):
                    ptypes.append(at.text)
            else:
                print('\t',pmid, 'no pub type')
            ptypes = '|'.join(ptypes)
            nodes = article.findall('Abstract')
            if( (len(nodesp) > 0) and (len(nodes) > 0) ):
                abstract = article.findall('Abstract')[0]
                for at in abstract.findall('AbstractText'):
                    lab = ''
                    if( 'Label' in at.attrib ):
                        lab = at.attrib['Label']
                    txt = str(at.text)
                    #try:
                    chex = set( list(re.findall('&#x([a-z0-9]+);', txt)) )
                    for it in chex:
                        try:
                            txt = txt.replace( f"&#x{it};", bytes.fromhex(str(it)).decode('utf-8') )
                        except:
                            pass
                    #except:
                    #    pass
                    lines.append( [ pmid, lab, ptypes, txt ] )
            else:
                print('\t',pmid, 'no abstract')
        
        return lines
        
    def get_abstracts(self):
        omap = os.path.join( self.out, 'abstract_info_pubmed.tsv' )
        gone = set()
        if( os.path.isfile(omap) ):
            fm = open( omap, 'r' )
            for line in fm:
                l = line.replace('\n','')
                if( (len(l) > 2) and (not l.startswith('pmid') ) ):
                    gone.add( int( l.split('\t')[0] ) )
            fm.close()
        else:
            fm = open( omap, 'w' )
            fm.write("pmid\tlabel\tpublication_type\ttext\n")
            fm.close()
    
        inmap = os.path.join( 'out', 'complete_mapping_pubmed.tsv' )
        df = pd.read_csv( inmap, sep='\t')
        allids = set(df['pmid'].values)
        df = df[ ~df['pmid'].isin(gone) ]
        ids = set(df['pmid'].values)
        print( 'Remaining: ', len(ids), '/', len(allids) )
        for id in tqdm(ids):
            try:
                lines = self._parse_paper( str(id) )
                if( len(lines) > 0 ):
                    lines = list( map( lambda x: ("\t".join(x)), lines ))
                    with open( omap, 'a' ) as fm:
                        fm.write( "\n".join(lines)+"\n" )
            except:
                print( 'error', id)
    
    def complete_publication_type(self):
        to_fix = set( ['12321046', '25821536', '28705208', '35521657', '31065384', '34867030', '31590286', '34473939', '31721601', '30935321', '30935324', '30935325', '34867586', '33425820', '34867626', '30542314', '30673508', '31984244', '34081487', '30804739', '30018558', '28314788', '30411986', '31329510', '33426938', '31985294', '30674610', '36179705', '28446496', '28446499', '33689454', '33296258', '31723443', '34869187', '32903360', '34869472', '34869534', '32117045', '28447069', '31199682', '32379496', '35263188', '28971851', '35787693', '34870194', '28316740', '30545067', '29234420', '36837442', '36837467', '36837572', '33561085', '28580433', '30415588', '30809168', '31464548', '30809328', '38673681', '32644454', '30809523', '30678531', '32644626', '30941095', '31858880', '31334667', '33170049', '32645830', '32121611', '30942129', '29631455', '27403491', '31204604', '35005701', '35136878', '23865096', '36317012', '30943119', '34744669', '33434107', '34744989', '31730370', '30026911', '26487977', '28978680', '30813769', '33042004', '36187849', '35925809', '33829030', '35532984', '34222272', '32125131', '28848465', '32256348', '30159284', '30421571', '36188784', '28848899', '30815001', '33960817', '33960833', '35796097', '33043697', '33568190', '35928033', '28588063', '28326159', '33306939', '35273031', '32520594', '31865758', '31865770', '36060611', '29769208', '28112825', '28458592', '35143273', '34356979', '27803501', '30687151', '34095079', '32129111', '28459105', '28066701', '39601043', '30950656', '28722845', '31213778', '32393479', '31213922', '39996125', '33835830', '28593015', '34753535', '31607967', '30821559', '30821630', '33049927', '30428518', '28724646', '28724657', '32525755', '33443618', '33050409', '31215818', '29250012', '28070420', '33706543', '30429768', '31216215', '28070521', '28201840', '38163380', '33576249', '32265700', '35935863', '28202639', '36067074', '36984590', '32528399', '29644891', '29644897', '25057539', '34232582', '32528650', '31611797', '31088186', '36200056', '35020462', '34496464', '30957581', '33841184', '31613140', '30958033', '28336780', '27812520', '32924568', '32793517', '35021890', '28861514', '22439157', '28469546', '32270694', '30698049', '37251792', '34761694', '31353964', '32271666', '28077381', '31354513', '28077732', '28077734', '34631809', '32665843', '32010701', '32010745', '35549708', '32404003', '30700232', '34239206', '31486723', '33715033', '30307351', '32142388', '32666871', '34240001', '34108929', '32667261', '38696672', '30832440', '30570358', '33322949', '32667837', '28473595', '33585418', '37910822', '32536979', '33585578', '30702005', '31095423', '28473993', '30964365', '33716945', '35420921', '31620067', '33192959', '29785237', '30702833', '30833915', '30965011', '28212588', '33587001', '26640293', '33849367', '32407638', '31752634', '35947087', '30704332', '33325799', '34636627', '33194943', '35030040', '35030070', '33326335', '35947804', '32802287', '29656669', '29263572', '35162346', '30705964', '33720749', '33720787', '29657667', '33459040', '30837835', '39226471', '36342964', '32411082', '30576221', '31231656', '36343655', '32411717', '32936046', '33329303', '33329324', '35295526', '35426735', '28480078', '32150360', '28611687', '30184559', '32412858', '30971922', '31496381', '35821806', '35821820', '29399304', '27695404', '36215171', '33724867', '34904920', '32545856', '35167779', '32153208', '35036831', '33464147', '32153497', '32940537', '33464993', '28615379', '31106141', '28878035', '31630566', '25077185', '34383533', '28747487', '27699253', '27699270', '32155789', '36350349', '33335875', '35695197', '31239736', '34123370', '30060300', '32812942', '28749772', '34385969', '32026743', '28487814', '34124111', '33730977', '29798906', '31502879', '29405811', '28619831', '31765604', '27571484', '32421353', '33207794', '32945728', '35960508', '33208055', '37533665', '32553024', '27965544', '34650509', '33601960', '34912869', '28097230', '28097231', '28490510', '28490514', '28228422', '28228424', '28228425', '28490674', '30325701', '35699655', '34520250', '28491040', '28360027', '34913844', '33996757', '33996843', '31244376', '33210466', '35045748', '31638448', '35570651', '31114211', '31114268', '28231596', '33998898', '28101105', '35572360', '35834620', '34524033', '33344496', '29543548', '34917633', '33345095', '35966574', '35049354', '29806553', '30724101', '31379553', '31379701', '31511230', '35574491', '39768803', '28496925', '33739874', '31118757', '31118843', '31249919', '29021762', '34789085', '32167657', '35968761', '30070803', '36493334', '32299254', '32823781', '34658904', '31644304', '31251090', '31251094', '31644380', '31382524', '33348644', '28761217', '32038757', '31514508', '34529503', '4645131', '29549159', '29942422', '40428877', '29025767', '33613422', '28239491', '28894899', '30992105', '34793244', '37676977', '31647774', '29944036', '31779508', '35187396', '35056370', '27847545', '31648670', '31648672', '28765329', '35712448', '31387410', '30994376', '35712977', '34926573', '29945920', '31780963', '34795626', '34795635', '31781021', '30863660', '29029555', '36238568', '32961834', '35714349', '32306852', '36763338', '29030354', '33093712', '29554842', '34797737', '27588932', '31914347', '31259250', '33356422', '30997357', '34143682', '28376549', '33488460', '35061577', '27721747', '34013254', '29950349', '31916448', '35324322', '28246488', '35980115', '33751912', '34931569', '31523745', '33096743', '33883176', '25101381', '35587146', '25363551', '34276449', '36373820', '31393222', '34670026', '31131085', '31262379', '31262378', '35194596', '31525129', '34277679', '36375115', '30608007', '36506564', '30346254', '29559834', '34671652', '33361097', '35196114', '34409850', '39521725', '30870983', '22220321', '34279169', '30216086', '28250195', '31264957', '28381530', '28512647', '27857693', '28119932', '33362896', '28251175', '34805058', '30872988', '34805199', '30873106', '29562394', '29955682', '29955707', '35198625', '28251936', '28251937', '12785461', '27990163', '32446821', '34150872', '30087760', '31922899', '28515069', '28515160', '34413804', '34283064', '37297840', '31137738', '27729953', '31662165', '27730189', '33890819', '29434463', '28779400', '35333140', '28255259', '34284613', '33760658', '32974255', '35727117', '28649296', '31401903', '31009235', '36252151', '33237689', '31402740', '34155899', '34287023', '33107400', '32190059', '32452284', '31928229', '30617563', '32583749', '32059566', '31404224', '31535597', '30880501', '30880522', '30618593', '25506953', '31143160', '30618929', '30618938', '33633656', '30619031', '27866528', '34551218', '30619077', '29046301', '29046302', '29046304', '30881353', '7026295', '36255448', '29046744', '31275098', '21969017', '31275206', '33110397', '32324065', '32324082', '31145045', '32849209', '33373537', '32193971', '36126298', '34816163', '33112646', '35734087', '32195186', '29311916', '32850876', '33375192', '32326796', '30622936', '32064853', '35079583', '28264221', '28395364', '28395367', '30230399', '34555822', '34818097', '31017022', '30624150', '35867052', '33114546', '26823383', '32983991', '30231535', '30756091', '14634268', '27872555', '29445521', '32722338', '31804851', '35081741', '29445691', '34033271', '33902320', '33902326', '34819911', '31019480', '32068113', '36524597', '31544065', '32068436', '35083238', '29447448', '37180729', '31282758', '33117961', '32462950', '30759059', '29448808', '33643204', '32594705', '34823095', '28138492', '30628860', '35085455', '29973911', '33643965', '33644099', '34692725', '32595586', '34692757', '29188066', '31809602', '11362451', '34431329', '31285919', '29319988', '28795872', '20669513', '33776891', '32990494', '31024430', '30631299', '35087875', '30369290', '34432541', '11363983', '27748093', '11364104', '34957244', '32991242', '11364377', '31681030', '28404577', '34434003', '31681579', '32206052', '34041090', '35876255', '36269661', '31813283', '28405623', '31944644', '36532230', '32338063', '28275125', '28275128', '28275132', '34566769', '32338562', '30896903', '28275532', '27227200', '33256540', '31552762', '30766492', '31815069', '28931532', '34698750', '32208570', '31553265', '30766862', '33650449', '30242790', '32208953', '32471194', '31422713', '28539413', '30505830', '33651671', '33651675', '34307246', '33521019', '34831767', '31293437', '23036135', '28279033', '33784486', '28148682', '29066505', '34309408', '29984358', '35096167', '34440974', '30246795', '35620899', '31951150', '29854097', '32083083', '34966795', '29461771', '36146637', '28282690', '29200450', '36147329', '35230142', '29856244', '29200944', '33395354', '33788611', '32609011', '35100025', '29070713', '30774790', '32216722', '29988503', '28546877', '35100697', '33659261', '31824412', '30251569', '31169077', '34708091', '35232533', '30907166', '29334322', '34970565', '28154824', '33398260', '34447303', '33661054', '33923278', '7053558', '31171071', '37200510', '29336228', '31433608', '30778345', '30647756', '31303149', '33269310', '32089783', '31434537', '30517043', '34449262', '34056067', '35235720', '33138623', '31959019', '29862060', '28420720', '36416175', '33270531', '28420943', '28814394', '28814395', '31304887', '33008942', '33140018', '33140020', '34713183', '31829732', '25276497', '34582721', '33272083', '31306043', '32223688', '34321144', '28685096', '29078743', '35108175', '33404325', '33535589', '28424185', '28293270', '29997428', '33274349', '30915273', '30915309', '32357109', '31046402', '33274648', '31308733', '33536998', '30522343', '31177727', '30522571', '32226774', '28819385', '32489718', '32882958', '31179027', '30785860', '32883018', '32096654', '31179244', '29999936', '36160428', '30786750', '33015104', '20301301', '30131727', '31311386', '20301411', '20301605', '34195237', '20301661', '28690539', '33409161', '32491937', '28428910', '30919704', '30395486', '31050927', '40094912', '32362168', '31313626', '31313942', '28954840', '33935686', '33935773', '35377567', '31314614', '31314680', '34722661', '24761346', '38917366', '33682556', '28694340', '33544066', '31971463', '33937603', '32757963', '31971890', '30136898', '30792343', '32234139', '27908967', '31448107', '31710277', '33283151', '31317799', '33021843', '36036829', '34595223', '31318603', '28959500', '32760739', '28566523', '28566529', '33023304', '30140207', '29354049', '30796037', '29485355', '20704057', '29092882', '30010440', '31976819', '35384736', '31190584', '31583951', '31190937', '31190975', '31191009', '34336894', '34205871', '34992327', '32764112', '32240037', '29356462', '28439135', '35254948', '35910354', '30405397', '30405414', '30405417', '32633688', '30405526', '35255220', '35255229', '33944698', '31847626', '29488407', '28702003', '33551718', '30406063', '33028076', '12318806', '27785374', '31193430', '34077032', '27261520', '28179223', '28179226', '32242498', '33553232', '34077678', '28179488', '34733185', '31456431', '31849809', '34471447', '28835364', '28835366'] )
        omap = os.path.join( self.out, 'abstract_info_pubmed.tsv' )
        fm = open( omap, 'r' )
        for line in fm:
            l = line.replace('\n','').split('\t')
            if( (len(l) > 1) and (not l[0].startswith('pmid') ) and ( l[2] == '' ) ):
                to_fix.add( int( l[0] ) )
        fm.close()
        
        pmap = os.path.join(self.out, 'fix_pubmed_abs.tsv')
        gone = set()
        if( os.path.isfile(pmap) ):
            fm = open( pmap, 'r' )
            for line in fm:
                l = line.replace('\n','')
                if( (len(l) > 2) and (not l.startswith('pmid') ) ):
                    gone.add( int( l.split('\t')[0] ) )
            fm.close()
        else:
            fm = open( pmap, 'w' )
            fm.write("pmid\tlabel\tpublication_type\ttext\n")
            fm.close()
        to_fix = to_fix - gone
        print('Fixing ', len(to_fix), 'articles data' )
        for id in tqdm(to_fix):
            try:
                lines = self._parse_paper( str(id) )
                if( len(lines) > 0 ):
                    lines = list( map( lambda x: ("\t".join(x)), lines ))
                    with open( pmap, 'a' ) as fm:
                        fm.write( "\n".join(lines)+"\n" )
            except:
                pass
                
        fixdf = pd.read_csv( pmap, sep='\t')
        inmap = os.path.join( self.out, 'abstract_info_pubmed.tsv' )
        df = pd.read_csv( inmap, sep='\t')
        df = df[ ~df['pmid'].isin(to_fix) ]
        df = pd.concat( [df, fixdf] )
        df = df.drop_duplicates()
        df = df.reset_index()
        df.to_csv(inmap, sep='\t', index=None)
    
    def get_grouped_abstracts(self):
        self.logger.info('--->Getting abstracts of articles associated to NCT raw info')
        omap = os.path.join( self.out, 'group_abstract_info_pubmed.tsv' )
        gone = set()
        if( os.path.isfile(omap) ):
            fm = open( omap, 'r' )
            for line in fm:
                l = line.replace('\n','')
                if( (len(l) > 2) and (not l.startswith('pmid') ) ):
                    gone.add( l.split('\t')[0] )
            fm.close()
        else:
            fm = open( omap, 'w' )
            fm.write("pmid\tlabel\tpublication_type\ttext\n")
            fm.close()
    
        inmap = os.path.join( self.out, 'complete_mapping_pubmed.tsv' )
        df = pd.read_csv( inmap, sep='\t')
        allids = set( [ str(id) for id in df['pmid'].unique() ] )
        ids = allids - gone
        interval = 500
        k = 0
        c = 1
        self.logger.info( f"\tIt will obtain {len(ids)}/{len(allids)}" )
        while (k < len(ids) ):
            rids = list(ids)[k:k+interval]
            self.logger.info( f"\tRetrieving and parsing chunk {k}/{len(ids)}" )
            ftmp = os.path.join( self.out, f'gr-{c}_tempFile.xml' )
            fetch = Entrez.efetch(db='pubmed', resetmode='text' ,id = (','.join(rids)), rettype='abstract')
            try:
                with open( ftmp, 'wb') as f:
                    f.write(fetch.read())
                try:
                    lines = self._parse_paper_group( ftmp )
                    if( len(lines) > 0 ):
                        lines = list( map( lambda x: ("\t".join(x)), lines ))
                        with open( omap, 'a' ) as fm:
                            fm.write( "\n".join(lines)+"\n" )
                except:
                    pass
            except:
                pass
            k += interval
            c+=1
    
    def generate_prediction_inputs(self):
        inmap = os.path.join( self.out, 'group_abstract_info_pubmed.tsv' )
        df = pd.read_csv( inmap, sep='\t')
        df = df[ ~df['text'].isna() ]
        all_pmids = len( df.pmid.unique())
        pmids_with_ctid = df[ df['text'].str.contains('NCT0') ].pmid.unique()
        df = df[ df.pmid.isin(pmids_with_ctid) ]
        print('All pmids', all_pmids )
        print('Pubmed ids with mapped ctid', len(df.pmid.unique()) )
        '''
All pmids 376984
Pubmed ids with mapped ctid 68718
Combinations pmid+label_abstract_piece : 320285
        '''

        '''
        ids = set( df.pmid.unique() )
        df = df[ df.text.str.contains('randomised') ]
        rctids = set( df.pmid.unique() )
        print( len(rctids), '/', len(ids), 'are randomized' )
        '''
        txts = {}
        for i in df.index:
            pmid = df.loc[i,'pmid']
            lab = str(df.loc[i, 'label']).replace(' ', '-').replace('/', '-').replace(',', '-').replace('nan', '')
            fname = f"{pmid}_{lab}.txt"
            #if( not fname in txts):
            #    txts[fname] = []
            text = df.loc[i, 'text']
            #txts[fname].append(text)
            txts[fname] = text.replace('"','').replace("'",'')
        
        for fname in tqdm(txts):
            #text = ' '.join( txts[fname] )
            text = txts[fname]
            opath = os.path.join( self.pred, fname )
            f = open(opath, 'w')
            f.write(text)
            f.close()
    
    def check_abstracts_mapped_coverage(self):
        omap = os.path.join( self.out, 'group_abstract_info_pubmed.tsv' )
        pgone = set()
        fm = open( omap, 'r' )
        for line in fm:
            l = line.replace('\n','')
            if( (len(l) > 2) and (not l.startswith('pmid') ) ):
                pgone.add( l.split('\t')[0] )
        fm.close()
    
        inmap = os.path.join( self.out, 'complete_mapping_pubmed.tsv' )
        df = pd.read_csv( inmap, sep='\t')
        callids = set( [ str(id) for id in df['ctid'].unique() ] )
        pallids = set( [ str(id) for id in df['pmid'].unique() ] )
        f = df[ df['pmid'].isin( [ int(i) for i in pgone ] ) ]
        ctf = set(f.ctid.unique())
        ids = pallids - pgone
        print( 'All mapped CTs', len(callids) )
        print( 'All mapped pubmeds', len(pallids) )
        print( 'Missing pubmeds', len(ids) )
        print( 'All cts with abstract parsed', len(ctf), ' - ', ( len(ctf)/len(callids) ) )
        print( 'All pubmeds with abstract parsed', len(pgone), ' - ', ( len(pgone)/len(pallids) ) )
        '''
        All mapped CTs 116603
        All mapped pubmeds 401674
        Missing pubmeds 21104
        All pubmeds with abstract parsed 380570  -  0.947
        '''
    
    def run(self):
        #self.get_abstracts()
        #self.complete_publication_type()
        
        self.get_grouped_abstracts()
        self.generate_prediction_inputs()
        
        self.check_abstracts_mapped_coverage()
        
if( __name__ == "__main__" ):
    odir = './out'
    i = ProcessPubmed( odir, predDir='/aloy/home/ymartins/match_clinical_trial/experiments/new_data/' )
    i.run()

