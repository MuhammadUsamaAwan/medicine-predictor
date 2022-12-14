const medicineList = [
  'SHARP POINT PHACO KNIFE',
  'ACETIN',
  'ABLE SPACER',
  'ACTRAPID HM (REGULAR) VIAL OF 10 ML',
  'ACTODERM',
  'ACURON',
  'ACYLEX',
  'ADALAT LA',
  'ADALAT RETARD',
  'ADENOSINE',
  'ADOLAN',
  'ADRENALINE',
  'ADRONIL',
  'ADVACORT',
  'ADVANT',
  'ADVANTAN',
  'AEROKAST',
  'AIRTAL',
  'ALBAZOLE',
  'ALBOPLEX',
  'ALCAIN 15ML',
  'ALDACTONE',
  'ALDOMET',
  'ANEXATE',
  'ANGIOCARD',
  'ANGISED',
  'AMINOPLASMAL E5 5%',
  'AMOXIL',
  'AMINOVEL -600',
  'AMITIN',
  'ANOROL',
  'AQUAZOLE CREAM',
  'AMDOXINE',
  'AMGYDEX',
  'ANOROL FORTE',
  'ANSAID',
  'AMLOCARD',
  'AMLOD',
  'AMORIN',
  'AMPRESS',
  'AMVAX-B',
  'AMYGRA',
  'APROVEL',
  'AMYLINE',
  'ANASTROZOLE SANDOZ',
  'AMDIPINE',
  'ALFA-D',
  'ALPHAGAN EYE',
  'AMARYL',
  'ALLERPHENE',
  'ALP',
  'ALPHAGAN',
  'AVIL',
  'ATEM',
  'AUGMENTIN',
  'AVELIN',
  'ATEM NEBULIZING NEBULES',
  'AVELOX',
  'ASCARD',
  'ARIXTRA',
  'ARTIFEN',
  'AROCAINE',
  'ARTEM DS',
  'ASCARD PLUS',
  'ASCORBIC ACID',
  'ARTERIAL SHEATH (6F)',
  'ARBI',
  'ARBI-D',
  'AREFED',
  'ARTHEGET DS',
  'ASUNRA',
  'ARIMEDIX',
  'ARINEC FORT',
  'BANDAGE ELASTO CRAPE 10 CM*4.5 MT',
  'BANDAGE ELASTO CRAPE 15 CM*4.5 MT',
  'BANDAGE COTTON',
  'BANDAGE COTTON 8cm * 4.5mtr',
  'BANDAGE TRIANGULAR',
  'BANDAGE COTTON (FOC)',
  'BASAGINE',
  'BEKSON FORTE',
  'BANDAGE COTTON 15 CM * 4.5 MTR 6"',
  'BENZIBIOTIC',
  'BANDAGE COTTON 2"',
  'BANDAGE COTTON 7.5 CM * 4.5 MTR 2*1/2',
  'BANDAGE POP GYPSONA 15cm*2.7m CM',
  'BACTIGRAS #7461',
  'AZAFRIN',
  'AZOMAX',
  'AZITMA 500MG',
  'BACLIN',
  'BACTROBAN',
  'BAIN ANAESTHESIA CIRCUT (ADULT)MAPLESON-D 2115',
  'BAIN BREATHING CIRCUIT ADULT 2105',
  'BALIGNO M2%',
  'AVODART',
  'BIKIL',
  'BEVIDOX',
  'BEPSAR',
  'BETADERM',
  'BICARBONATE HAEMODIALYSIS CONCENTRATE SOLUTION A+B',
  'BIGUACID PLUSE DISINFACTANT',
  'BETAGENIC',
  'BETALOL EYE DROP 0.5%',
  'BETAMET LOTION',
  'BETACIN',
  'BETATEK EYE',
  'BETNELAN',
  'BETNESOL',
  'BETNESOL EYE',
  'BETNOVATE',
  'BREATHING/SYSTEM COMPACT 2154(2 MTR +2LTR BAG +1.5 MTR LIMB) ADULT',
  'BRONIL',
  'BOFALGAN',
  'BREXIN',
  'BRITANYL',
  'BROTIN',
  'BRITANYL EXP',
  'BONEFOS',
  'BONJELA',
  'BONVIVA',
  'BOSMON',
  'BROMYCINE-D EYE',
  'BRONCHILATE',
  'BLOTIM EYE 5ML',
  'BLOKIUM DIU',
  'BLEPHAMIDE EYE',
  'BLEPHAPRED EYE',
  'BLINK FRESH 10ML',
  'BLOKIUM',
  'BLADE SURGICAL SIZE-11 CARBON STEEL',
  'CALCIUM LACTATE',
  'CALPOL',
  'CALAMOX',
  'CALAN',
  'CALCIUM GLUCONATE',
  'CALCIUM-LACTATE',
  'CALFIT',
  'CALSUP',
  'CALAN SR',
  'BRUFEN',
  'C.V.P LINE SIZE 16  20 CM',
  'BUNILAX',
  'BUPICAIN',
  'BUPICAIN 10ML',
  'C.V.P LINE SIZE 18  20 CM',
  'C.V.P LINE SIZE G-20  20 CM',
  'BUPICAIN SPINAL 2ML',
  'BYNEVOL',
  'BYSCARD',
  'CAC (EFFERVESCENT)',
  'CAC PLUS 1000',
  'CAC-1000 PLUS',
  'CAFLAM',
  'C-PHOS',
  'C.V.P LINE SIZE 14  20 CM',
  'CANNULA IV S-22  WITH INJ.PORT',
  'CANNULA NASAL OXYGEN',
  'CANNULA IV S-20 WITH INJ.PORT',
  'CAPECITABINE NORMON',
  'CAPEGUARD',
  'CAPOTEN',
  'CANNULA I.V G. 24 BRANNULE WITH STOPER',
  'CANNULA IV S-18  WITH INJ.PORT',
  'CALZEM',
  'CANNULA IV G-16 WITH INJ.PORT',
  'CATHETER NELTON S-16',
  'CATHETER NELTON S-18',
  'CATHETER SILICONE SIZE 16 FR',
  'CATHETER NELTON S-8',
  'CATHETER NELTON SIZE 10',
  'CATHETER AERO FLOW TIP SUCTION',
  'CATHETER NELTON SIZE 14',
  'CATHETER SILICONE SIZE 18 FR',
  'CATHETER NELTON SIZE 16',
  'CATHETER BRONCHO (DOUBLE LUMEN TUBE) - LEFT',
  'CATHETER NELTON SIZE 18',
  'CATHETER NELTON SIZE 20',
  'CATHETER BRONCHO (DOUBLE LUMEN TUBE) RT',
  'CATHETER BRONCHO DOUBLE LUMEN',
  'CATHETER BRONCHO DOUBLE LUMEN - LEFT',
  'CATHETER BRONCHO DOUBLE LUMEN - RIGHT',
  'CATHETER NELTON SIZE 8',
  'CARVEDA',
  'CATHETER LONG TERM DIALYSIS((PERMACATH)',
  'CATHETER SILICON SIZE 12 FR',
  'CATHETER SILICONE SIZE 10 FR',
  'CATHETER SILICONE SIZE 14 FR',
  'CATHER WESTREN (EXTERNAL,LARGE,MED)',
  'CATHETER MOUNT ADULT',
  'CATHETER NELTON S-12',
  'CATHETER NELTON S-14',
  'CARICEF',
  'CAPRIL',
  'CARDNIT',
  'CARPRO',
  'CARSEL',
  'CAPTIL',
  'CARDURA',
  'CARTIGEN PLUS',
  'CARA',
  'CITANEW',
  'CITOLIN',
  'CIPROXIN',
  'CIPOL-N SOFT',
  'CITOLINE',
  'CIPOTIC EAR',
  'CITRO SODA (SACHET)',
  'CLENIL-A',
  'CITROSODA',
  'CIPRALEX',
  'CILAPEN',
  'CLARITEK',
  'CELL CULTURE RABIES VACCINE',
  'CHYMORAL',
  'CECLOR',
  'CEFAMAX',
  'CEFAMEZIN I/V',
  'CEFCOM',
  'CEFIPIME',
  'CELLCEPT',
  'CEFRASUL',
  'CATHETER SILICONE SIZE 6 FR',
  'CATHETER SILICONE SIZE 8 FR',
  'CATHETER TROCAR (C/TUBE) S-24',
  'CECON',
  'CHEST DRAINAGE BOTTLE',
  'CEBAC',
  'CELBEX',
  'CHLOROPHENRAMINE',
  'CECLOFIN',
  'CHLOROQUINE',
  'CHLORPHENIRAMINE',
  'COLOSTOMY BAG 15*25CM-51MM',
  'CONCOR',
  'CO EZIDAY',
  'CO-DIOVAN',
  'CO-DIOVAN 160/12.5 MG',
  'CO-DORZOL',
  'CO-DORZOL OPTHALMIC SOLUTION',
  'CLOZOX VAG',
  'CO-RENITEC',
  'CO-SOPT OTHALMIC',
  'COLISTIM',
  'CLOZOX VAG.',
  'COMFEEL DRESSING',
  'CLYCIN T',
  'CO METHER',
  'CO METHER  (60ML)',
  'CLYCIN-V VAGINAL',
  'CO TRUPRIL',
  'CO-APROVEL',
  'COLOPLAST OSTOMY BELT',
  'COLOSPAS',
  'CLOBEDERM',
  'CLOTRIM',
  'CLENIL-A-NEBULES',
  'CLEXANE',
  'CLOBICARE',
  'CLOVIR',
  'CLOMFRANIL',
  'CLOZARIL',
  'CLOPIXOL ACCUPHASE',
  'CLOPIXOL DEPOT',
  'CLOPRA',
  'CLOTLES SS',
  'CLINICA',
  'CLOTNIL',
  'DANZEN DS',
  'CREMAFFIN',
  'CROMOG EYE 2%',
  'CYROCIN',
  'Citro Soda',
  'D-TRES',
  'DANZOL',
  'CURE-C',
  'CUTTER FOR NEURO SURGICAL',
  'CYCLOPEN EYE',
  'DAFLON',
  'DAKTARIN',
  'CYCLOSEN',
  'CYCLOZ',
  'CYNFO',
  'CRAFILM',
  'CORDARONE',
  'CONVATEC COLOSTOMY POUCH SIZE 70MMASSRTD',
  'CONVATEC SURFIT FLANGE 70MMASSRTD',
  'CONVATEC SURFIT TAIL CLOSURE / POUCH CLAMP',
  'CORALAN',
  'CORTICORT',
  'CORTISPORIN EYE',
  'COVA',
  'COVA-H',
  'COTTON WOOL ABSORBANT400 G',
  'COVERSYL',
  'CR RAXIL',
  'DEX 25% IN WATER',
  'DEVOM',
  'DEX 25% WATER',
  'DEX.5% IN WATER',
  'DEPRICAP',
  'DERMOSPORIN',
  'DERMOVATE',
  'DEX.10% IN WATER',
  'DEX.5% IN SALINE + NACL',
  'DESFERAL',
  'DEX 10% IN WATER',
  'DELTACORTIL EC',
  'DEPOMEDROL',
  'DECADRAN',
  'DEBRIDATE',
  'DEBRITONE',
  'DEPREL',
  'DENLA XR',
  'DAPA',
  'DECON-A',
  'DAPWIZ',
  'DEPLAT',
  'DELTACORTIL',
  'DIGNITY SHEET 75 GM',
  'DIMECO',
  'DISPOSABLE GLOVES  PLOYTHENE',
  'DINEMIC SR',
  'DIJEX MP',
  'DICLORAN',
  'DIOVAN',
  'DICLORAN SR',
  'DIFLUCAN',
  'DIGITEK',
  'DISPIRIN CV',
  'DEXAMETHASONE',
  'DEXATOB EAR',
  'DIABOLD',
  'DIAMICRON',
  'DIAMICRON MR',
  'DEXA',
  'DIANE-35',
  'DIATHERMY LEAD',
  'DULAN',
  'DOPAMINE',
  'DOSIK',
  'DORMICUM',
  'DOUBLE LUMEN,SUB CLAVIAN CATH ADULT',
  'DUODART',
  'DOMEL',
  'DRATE',
  'DOSI FLOW (DI-LA-FLOW)',
  'DRENOVAC BOTTLE WITH CATHETER (REDON DRAIN BOTT WITH CATHETER)/PRIVAC',
  'DISPROL',
  'DISTILLED WATER',
  'DISPOSABLE SURG GOWNS',
  'DIU-TANSIN',
  'DIZET DS',
  'DISPRIN',
  'DOBUTAMINE',
  'ENDOTRACHEAL TUBE (WITH CUFF) 6.5',
  'ENFLORE SACHAT',
  'ELUSEN',
  'EMG NEEDLE',
  'ENIER',
  'EPINOR',
  'ENDOTRACHEAL TUBE (WITH CUFF) 7.0',
  'ENDOTRACHEAL TUBE (WITH CUFF) 7.5',
  'ENTAMIZOLE',
  'EFEXOR XR',
  'ENTAMIZOLE DS',
  'ENDOTRACHEAL TUBE (WITH CUFF) 6.0',
  'EPIDURAL SET (ANAESTHESIA UNIT)G 16',
  'EPIDURAL SET(ANAESTHESTA UNIT)G-18',
  'DUPHASTON',
  'DYNAQUIN',
  'ECHELON 45 RELOAD REGULAR',
  'EBSTINE',
  'ECASIL',
  'DYCLO',
  'DYCLO SR',
  'DUODERM DRESSING 4*4 / RESTORE PLUS',
  'ECAVIR',
  'DYMIN',
  'DUODERM DRESSING 8*8 RESTORE PLUS',
  'DUOFILM',
  'DUPHALAC',
  'EZIDAY',
  'FEEDING TUBE S 10',
  'EVOPRIDE',
  'EVOROX',
  'EXTENSION TUBES (JMS)',
  'EVOTIN',
  'EYEBREX',
  'EYFEM',
  'FEEDING TUBE S 16',
  'EVOZID',
  'EUGLUCON',
  'FACE MASK DISPOSABLE(3 LAYERED TAILORED 80GM))',
  'FASTUM GEL',
  'EVION',
  'FAVERIN',
  'EPURAM',
  'ETHICROM T',
  'EPIVAL',
  'ETHOMID',
  'ETIVIL',
  'EPIVAL- CR',
  'EPOKINE',
  'ESONEXT',
  'ESPRAMCIT',
  'ESSO',
  'ESTAR',
  'ESTEZENE',
  'EPOTIV  PREFILLED',
  'ETHICROM 10ML',
  'FOLLEY,S CATHETER 2 WAY SIZE 12 FR',
  'FOLLEY,S CATHETER 2 WAY SIZE 16 FR',
  'FLUANXOL DEPOT',
  'FOLIC ACID',
  'FLUCATE',
  'FLUDERM',
  'FOLLEY,S CATHETER 2 WAY SIZE 14 FR',
  'FLAGYL',
  'FLEXIN',
  'FLUROCORT',
  'FLUX',
  'FOLLEY,S CATHETER 2 WAY SIZE 10 FR',
  'FLIXONASE',
  'FML FORT',
  'FOCIN',
  'FEXET',
  'FEEDING TUBE S 7',
  'FEFAN',
  'FEEDING TUBE S 8',
  'FEFOL VIT',
  'FEFOL-VIT',
  'FIBRIN GLUE',
  'FILGEN',
  'FELDENE DISPERSIBLE',
  'FENOGET',
  'FERINIL',
  'FERROUS SULPHATE',
  'FEEDING TUBE S 6',
  'GAUZE SURGICAL 36"*40 MTR (FREE OF COST)',
  'GAUZE SURGICAL 36"*40 MTR - 28 PICS',
  'FUCIDIN',
  'FUSIMED-B',
  'G-MIDE',
  'GABICA',
  'GABIX',
  'FRUSEMIDE',
  'GABLIN',
  'GALVUS',
  'FUDIC',
  'GALVUS MET',
  'FUCICORT',
  'GALVUS- MET 58/1000',
  'GASTROGRAFIN LIQUID',
  'FROBEN',
  'FOLLEY,S CATHETER 2 WAY SIZE 18 FR',
  'FORTEXONE',
  'FOLLEY,S CATHETER 2 WAY SIZE 20 FR',
  'FOLLEY,S CATHETER 2 WAY SIZE 22 FR',
  'FOLLEY,S CATHETER 2 WAY SIZE 24 FR',
  'FORTUM',
  'FOSFOMYCIN',
  'FOSINE',
  'FOSTER INHALER',
  'FOTIFLOX EYE DROP(0.5%)',
  'FRED HOLLOWS IOL  6MM ASSORTED',
  'FORMIGET',
  'FORMIGET 12MCG',
  'GLOREX',
  'GIXER',
  'GLICON',
  'GLINEXT MR',
  'GENTAMYCIN EYE',
  'GENTICYN',
  'GENTICYN EAR',
  'GENURIN FORTE',
  'GETRYL',
  'GELAFUNDIN',
  'GELOFUSINE (GELATIN)',
  'GEMPID',
  'GLUCOPHAGE',
  'GRASIL',
  'GLUCOSTRIPS (VIVA CHECK)',
  'GRAVINATE',
  'GLOVES OPERATION SIZE 8',
  'GLUCANTIME',
  'GLUCOBAY',
  'GLUCOSTRIPS',
  'GOURIC',
  'GLOVES OPERATION SIZE 7',
  'GLOVES OPERATION SIZE 7.5',
  'HY CORTISONE',
  'HY-SOLONE',
  'HYDRALLAZINE',
  'HILIN',
  'HUMAN ALBUMEN',
  'HYDRINE',
  'HUMAN ALBUMINE',
  'HYDRO ACTIVE GEL',
  'HITOP',
  'HUMULIN-N',
  'HME FILTER (ADULT)',
  'HME FILTER (PAEDS)',
  'HOLLOW FIBER DIALYZIER WITH SET',
  'HOLLOW FIBRE DIALYZER 1.3 WITH SET',
  'HEPA-MERZ',
  'HERBESSER',
  'HERBESSER SR',
  'GREEN CARTRIDGE (TCR 75) (JOHNSON)',
  'GYTRIM - HC',
  'HAEMACCEL (POLYGELINE)',
  'HCQ',
  'HEPARIN',
  'HIGH CONCENTRATION OXYGEN MASK ADULT',
  'HAND SANITIZER',
  'HEBERBIOVAC',
  'IV GIVING  SET',
  'INSULATARD',
  'ISOPEARL GEL',
  'ISOTIN',
  'IVERMITE',
  'INVOCLOR EYE 10 ML',
  'INDROP ALPHA',
  'INDROP D',
  'INFLAMATIX',
  'ISMO',
  'INHIBITOL',
  'ISO PEARL',
  'INOGRAF',
  'ISOFLURANE',
  'ISOPEARL',
  'INDERAL',
  'HYDRYLLINE',
  'HYZONATE',
  'IMIDOL',
  'HYTRIN',
  'INDOMETHACIN',
  'IBANDRO',
  'IBERET',
  'HYDROCORTISONE',
  'HYDROCORTISONE 1%',
  'ICON',
  'ICON CAP',
  'IMATET 0.5 ML',
  'LAXOLAX',
  'LAME',
  'LAMICTAL',
  'LASIX',
  'LEFLOX',
  'LAMISIL',
  'LANOXIN',
  'LASIX 2ML',
  'LASORIDE',
  'KLEEN ENEMA',
  'LACOLIT',
  'LACRILUBE',
  'LATEP EYE',
  'KLINT',
  'LANSOLIB',
  'LATMO EYE',
  'LALAP',
  'LAXIDE',
  'LANTUS',
  'LANTUS PREFILLED PEN',
  'KOATE',
  'LANTUS SOLOSTAR',
  'LANTUS VIAL',
  'KLARACID',
  'KALGINATE DRESSING',
  'KINZ',
  'KAMPRO',
  'KEFZOLE',
  'KELFA',
  'KELFER',
  'KEMADRIN',
  'K N 95 MASK',
  'KENALOG ORABASE OINT',
  'KESTINE',
  'KITAZOO',
  'KETROSAN EYE',
  'KIDOCARB',
  'K-LOT',
  'LIPEREX',
  'LIPIGET',
  'LIGNOCAIN',
  'LILAC',
  'LEXOTANIL',
  'LERACE',
  'LEVEMIR',
  'LEXOF',
  'LEFORA',
  'LEPINZA',
  'MECOBAL 500MCG',
  'MECTIMITE',
  'MEPORE 9 X 30 CM',
  'MERCAPRINE',
  'MAGNESIUM SULPHATE',
  'MAXNA',
  'MAGNETT',
  'MEREM',
  'MANNITOL 20%',
  'MAXOLON',
  'MECOBAL',
  'MEFANAC',
  'LOWPLAT PLUS',
  'LOXAT',
  'MEFANAC DS',
  'LUCAST',
  'LUMARK',
  'MASACOL',
  'MEPORE 9 X 10 CM',
  'MEPORE 9 X 15 CM',
  'MEPORE 9 X 20 CM.',
  'MAX FLOW',
  'MAXIMA',
  'MEPORE 9 X 25 CM',
  'LOPROT',
  'LOTRIX',
  'LOWPLAT',
  'LISKOPLEX',
  'LOJIN',
  'LOPHOS',
  'LISINOLIB',
  'LOPRIN',
  'LOPROT I.V',
  'LORIP',
  'MIXTARD 70/30 HM 100',
  'MONIS',
  'MONITOR',
  'METROZINE',
  'METROZINE SUSPENSION',
  'METPHAGE',
  'MEXAIR',
  'MIACALCIC NASAL',
  'MICRONEMA',
  'MIGRAM',
  'MOFEST',
  'METOCLON',
  'MERONEM',
  'MEROID',
  'MEROL',
  'MERPEN',
  'MYSTATE',
  'MYRIN P',
  'NEBCIN',
  'MYCONIL',
  'NEBIL',
  'NEBRA',
  'MYDREX',
  'MYRIN P FORTE',
  'MYTEKA',
  'MYDRIACYL',
  'MYDROMIDE',
  'MOXIGET',
  'MYFORTIC',
  'MUCOLATOR',
  'MYFOTIC',
  'NALBIN',
  'MYRIN',
  'NATASAN EYE',
  'NATASAN EYE DROP',
  'NATRILIX SR',
  'MOVAX',
  'MONTIGET',
  'MOVELAT',
  'MONO REPID ALCOHOLIC LIQ/HAND WASH READY FOR USE',
  'MOVERYL',
  'MONOPTY GUN NO. 18',
  'MOSEGOR',
  'MOXIGAN',
  'MOTILLIUM',
  'NEUBEROL FORTE',
  'NEUROBION',
  'NEOPROX',
  'NEUBEROL',
  'NEOPYROLATE',
  'NEOSEGOR',
  'NEOMERCAZOLE',
  'NEUDOPA',
  'NERVIN',
  'NEOPHAGE',
  'NESTIGIN',
  'NEODIPAR',
  'NEBULIZER ADULT MASK',
  'NEEDLE SCALP VEIN (BUTTER FLY)G.25',
  'NEEDLE SCALP VEIN (BUTTERFLY) G 20',
  'NEEDLE SCALP VEIN (BUTTERFLY) G-19',
  'NEOGAB',
  'NEBULIZER PAED MASK',
  'NEEDLE SPINAL(LUM.PUNCTURE) G-23',
  'NEEDLE SCALP VEIN (BUTTER FLY)G.21',
  'NEEDLE SCALP VEIN (BUTTER FLY)G.22',
  'NEEDLE SCALP VEIN (BUTTER FLY)G.23',
  'NEEDLE TRUCUT BIOPSY ABC "6,"4.5',
  'NEEDLE SCALP VEIN (BUTTER FLY)G.24',
  'NEEDLE TRUCUT BIOPSY TRAVENOL"6,"4.5.',
  'NEO-K',
  'NORVASC',
  'NORMAL SALINE 0.9%',
  'NOVIDAT',
  'NOVAPRESSIN',
  'NOVIA (FOC)',
  'NORMAL SALINE0.9%',
  'NORMAL SALINE',
  'NEZKIL',
  'NEZKIL 60ML',
  'NICOTRIM',
  'NOGERD',
  'NOLVADEX',
  'NICOTRIM FORTE',
  'NIDOPIN',
  'NEUROLITH SR',
  'NEUROSTRESS',
  'NIFINE C.C',
  'NEXUM',
  'NISE',
  'NITROMINT SR',
  'NITRONAL',
  'OXYGEN MASK BLB DISPOSABLE ADULT',
  'OXYTETRACYCLIN',
  'OSSOBON-D',
  'OXIDIL',
  'PAED ADMINISTRATION SET',
  'OSTEOPOR OW',
  'OSTIM',
  'OPTRA INHALOR',
  'ORAL REHYDERATION SALT',
  'OTOFLOX EAR',
  'OXYGEN MASK-PAEDS WITH OXYGEN TUBE(CODE 1146)',
  'ORS (LOW OSMOLAR ORS)',
  'ORS (ORAL REHYDRATION SALT)',
  'ORSAL (LOW OSMOLAR)',
  'OSIRIS',
  'OSNATE - D',
  'OSNATE-D',
  'ONSET',
  'OPTOFLOX EYE',
  'OCUFLUR EYE',
  'OLANZIA',
  'OMECAP',
  'OPSITE 10 X 14',
  'OPSITE 45 X 55 CM/STERI DRAPE 2 60 X 45',
  'OPSITE 55 CMX45 CM',
  'OPSITE- MEDIUM SIZE (28CM X 15CM)',
  'NOVO FINE NEEDLES',
  'NOVOPHENICOL',
  'NUVAL D',
  'O-ZIP',
  'OMECER',
  'OMESEC',
  'OPSITESTERI-DRAPE 28 X 30 CM',
  'OPTACHLOR EYE',
  'OMNITOR',
  'ONDAMEX',
  'ONITA SACHET',
  'PERSCH',
  'PINE',
  'PAPER MASK DISPOSABLE',
  'PENFEN',
  'PIOMET',
  'PENRO',
  'PENTASA',
  'PIOZER',
  'PHARMATREXATE',
  'PIOZER PLUS',
  'PEPTIA',
  'PEPTICURE',
  'PIOZONE',
  'PERIDOL',
  'PIOZONE PLUS',
  'PARAXYL',
  'PHENERGAN',
  'PIDOGREL AP',
  'PARAXYL CR',
  'PANTRA',
  'PAEDIATRIC ADMINISTRATION SET',
  'PANADOL',
  'PONSTAN',
  'POTASSIUM CHLORIDE',
  'POLYFAX SKIN',
  'PONSTAN FORTE',
  'PLETAL',
  'POZEMET',
  'PLASTER MICROPORE (LEUKPOOR) 2" X 5 MTR',
  'POFOL',
  'POFOL 20 ML',
  'POT CHLORIDE',
  'PK-MERZ',
  'PLASBUMIN',
  'PIRIDE',
  'PLASTER ADHESIVE ZINC OXIDE 7.5 CM X 10 M',
  'PIRITON',
  'PYODINE SOLUTION 450 ML BOTT.',
  'PULMONOL',
  'PURIDONE',
  'PURINETONE',
  'PURPAL',
  'PYODINE SCRUB',
  'PROTHIADEN',
  'PREDNISOLONE',
  'PRIMOLUT-N',
  'PROCARBIZOLE',
  'PROCHOLIDIN',
  'PRED FORTE EYE DROPS',
  'PREFRIN A EYE',
  'PROLINE MESH 30*30',
  'PRELIN',
  'PROGYLUTON',
  'PROLUTON DEPOT',
  'PREMARIN',
  'PREMIUM PASTE 202/ADAPT PASTE/STOMAHESIVE PASTE',
  'PREDNISOLON',
  'PROPOFOL LIPURO',
  'PROSCAR',
  'PROSTREAT',
  'PROTAMINE SULPHATE',
  'PROTECT',
  'RHULEF',
  'RIBAGENE FREE OF COST',
  'RIBAVIRIN (CAP RIBAZOLE) FOC',
  'RELISPA',
  'RELTUS DM',
  'RIFADIN',
  'RIFAPIN-H',
  'RINGER LACTATE',
  'RENAVEL',
  'RIFATOL',
  'RENITEC',
  'RINO CLENIL NASAL',
  'RISEK',
  'RIFINAH',
  'REIN',
  'RING PESSARIES',
  'RING PESSARIES ALL SIZE 74',
  'RING PESSARIES ALL SIZE 80',
  'RING PESSARIES ALL SIZE 89',
  'RELOCURIUM',
  'RESOCHIN',
  'RESTORIL',
  'RELTUS C&F',
  'RANULCID',
  'REDEEM',
  'QALSAN-D',
  'REDON DRAIN TUBE',
  'RAMARGON',
  'RAZE',
  'RECOL',
  'QUENCH',
  'QUIBRON-T SR',
  'REFOBACIN',
  'RAMIPACE',
  'RANAX',
  'QUSEL',
  'RECORMON',
  'RANEY SCALP CLIP',
  'PYRAZINAMIDE',
  'RAE ENDOTRACHEAL TUBE (NON CUFF) 5.0',
  'QALSAN',
  'SANTODEX 0.1 %',
  'SCABIDERM',
  'SCABION',
  'ROVISTA',
  'SALBO',
  'SANTODEX OINTMENT',
  'SALMICORT INHALER',
  'SCABION 60 ML',
  'SALTRA',
  'SCHAZOBUTOL',
  'SEDIL',
  'SANDIMMUN (NEORAL)',
  'ROTEC',
  'SANTOVIR EYE 4.5 GM',
  'SARTAN H',
  'SARTAN H 50/12.25MG',
  'SCAB FREE',
  'ROVATOR',
  'SANTE',
  'S-SIGNIA FAST P/ REXTION ARENA PI(HEARING DEVICE)',
  'SALAZODINE',
  'SALAZODINE EC',
  'ROCEPHIN IV',
  'RIVOTRIL',
  'ROCONIUM',
  'ROCEPHIN IM/IV',
  'RONIROLE',
  'ROSUNEXT',
  'SOLUMEDROL',
  'SKILAX',
  'SKIN TACT ECG ELECTRODES-F60',
  'SOMOGEL',
  'SODIUM HYPOCHLORIDE CONC',
  'SOWEL',
  'SKIN STAPPLER',
  'SODAMINT',
  'SOFIGET',
  'SINAXAMOL EXTRA',
  'SOFOHIL',
  'SOFTIN',
  'SINEDOPA',
  'SINEMET',
  'SOFVASC',
  'SOL-CART B',
  'SOLIFEN',
  'SINGULAIR',
  'SITAGLU MET',
  'SIVAB',
  'SEPTRAN(PAEDS)',
  'SERADEP',
  'SERC',
  'SERLIN',
  'SERRITIDE',
  'SIGNIA FAST SP / SIGNIA FUN SP/REXTON ARENA HP-3(HEARING DEVICE)',
  'SILIMARIN',
  'SILIVER',
  'SERENACE',
  'SEIZUNIL',
  'SERT',
  'SEPTRAN',
  'SEPTRAN DS',
  'STERIFLUID NS',
  'STERIFLUID RL',
  'STERILLIUM',
  'STERIFLUTOL',
  'SPIDAR',
  'SPIROMIDE',
  'STERIFLUID 10%',
  'SSD 1% CREAM',
  'SPIROMETER',
  'STEMETIL',
  'STERIFLUID DS 1/2',
  'SPASLER P',
  'STERI STREP 1.5" *4"',
  'STERI STRIP 1/2"',
  'SPASMONIL',
  'STERIFLUID DS',
  'SYRINGES 10 ML (DISPOSABLE)',
  'SYRINGES 20 ML (DISPOSABLE)',
  'SYRINGES 3 ML ( AUTO DISABLE)',
  'SUSTAC',
  'SYNALGO',
  'SYNGAB',
  'SUCTION TUBE FOR SUCTION MACHINE',
  'SULPHAPRED EYE',
  'STOP CORK SPINAL NEEDLE 3 WAY WITH EXT TUBE',
  'SUPRAMYCIN',
  'SULZONE',
  'STIRUP',
  'STODIUM',
  'STOP CORK SPINAL NEEDLE 3WAY',
  'SUCFATE',
  'SUNI PLAST',
  'SUCTION CONNECTING TUBE 2 X CONNECTOR',
  'SUCTION NASAL YANKER',
  'SUPPOSITORY GLYCERINE  (INFANT)',
  'SUPPOSITORY GLYCERINE (ADULT)',
  'SYRINGES INSULIN 100 UNITS',
  'SYRINGES 60 ML CATHETERTIP',
  'TA-30 DIA 3.5 MM (STAPPLER) JOHNSON',
  'TACAVIR',
  'TAMSOLIN',
  'SYRINGES 50ML FOR INJ',
  'SYRINGES DISPOSABLE 20 ML CATHETERTIP',
  'SYRINGES 5 ML (AUTO DISPOSABLE)',
  'TEPH-20',
  'TESPRAL',
  'TERBIN',
  'TENDEM SKIN BERRIER 70MM',
  'TERBISAN',
  'TERNELIN',
  'TENOFO B',
  'TENORMIN',
  'TEROL',
  'TEGRAL',
  'TENDEM DRAINALBE POUCH',
  'TANSIN',
  'TEARS NATURALE II 15ML',
  'TENDEM SKIN BARRIER',
  'TENDEM SKIN BARRIER  45 MM',
  'TENDEM SKIN BERRIER 38 MM',
  'TENDEM SKIN BERRIER 57 MM (FLANGE)',
  'TARGOCID',
  'TED COMPRESSION STOCKINGS (PAIRS) ABOVE KNEE',
  'TEMO ERIGEN',
  'TEMOSIDE',
  'TARISIN',
  'TENDAM SKIN BARRIER 44 MM',
  'TENDEM DRAINABLE POUCH',
  'TENDEM DRAINABLE POUCH 38MM',
  'TENDEM DRAINABLE POUCH 45MM',
  'TAMSOLIN PLUS',
  'TENDEM DRAINABLE POUCH 55 MM',
  'TANDEM SKIN BERRIER 55 MM',
  'TENDEM DRAINABLE POUCH 57MM',
  'TAZOCIN',
  'TRAMAL',
  'TRANSAMIN',
  'TORADOL',
  'TOVIR',
  'TRACHEOSTOMY TUBE (WITH CUFF) NO 6',
  'TRACHEOSTOMY TUBE (WITH CUFF) NO 7',
  'TITAN IV/IM',
  'TIXYLIX',
  'TRACHEOSTOMY TUBE (WITH CUFF) SIZE 6.5',
  'TOBRACIN',
  'TRACHEOSTOMY TUBE( N/CUFF) NO. 8.5',
  'TRACNESAN CREAM',
  'TRADOL',
  'TOBRADEX EYE',
  'TOBREX',
  'TOCINOX',
  'TOFRANIL',
  'TOMAX',
  'TONOFLEX-P',
  'THYROXINE',
  'THYROXINE SODIUM',
  'TESTOVIRON DEPOT',
  'TIBOL',
  'THALIMID',
  'THEOGRAD',
  'TIENAM',
  'THIOLAX',
  'TIOVAIR',
  'ULCENIL',
  'TRITON',
  'TRONOLANE',
  'ULSANIC',
  'URINE COLLECTING (ADULTS) TYPE-A2000 ML',
  'TRUPRIL',
  'TYGACIL',
  'ULTRAVISIT 370',
  'UMBLICAL-CORD CLAMP',
  'TRISIL PLUS',
  'UNITREXATE',
  'TRAVOCORT',
  'TRIKAT MR',
  'TRANSPORE',
  'TRAZOLAM',
  'TRIOPTAL',
  'TRES ORIX FORTE',
  'TREVIA',
  'TREVIAMET',
  'TRIAZOLIN',
  'TRICARDIN',
  'VERICEF',
  'VIBRAMYCIN',
  'VASTEREL MR',
  'VENTOLIN',
  'VEDICAR',
  'VICERYL PLUS 3/0 26 MM 1/2 CRB (VCP 316 H)',
  'VEGAMAX EYE DROP 5%',
  'VALIUM',
  'VELOSEF',
  'VERMOX',
  'VANCOMYCIN',
  'VELPAGET',
  'VENALAX SR',
  'VENTOLIN INHALER',
  'VENTOLIN INHALOR',
  'VEREDET',
  'VEZITIC',
  'VENOFER 5ML',
  'VANCOMYCIN (FOC)',
  'URIXIN',
  'URSO',
  'UTORAL',
  'VAGIBACT CR',
  'URINE COLLECTING BAG (BABIES)',
  'WINTOGENO',
  'XADINE',
  'XALTIDE',
  'VOXAMINE',
  'WARFIN',
  'XALTIDE INHALOR',
  'VOLTRAL',
  'VISTAMIN',
  'VIT A&D',
  'VIT B COMPOUND',
  'VIT B-COMPOUND',
  'VOLTRAL SR',
  'VOMILUX',
  'WELCODOX',
  'VOREN',
  'X-PLENDED',
  'VORIF',
  'VINJEC',
  'VIGLIP-M',
  'VISLAT EYE DROPS',
  'VICTRIN',
  'VISRAL',
  'VIGLIP',
  'ZESTRIL',
  'ZIAPINE',
  'ZANTAC',
  'ZENTEL',
  'ZAPROTIN-20',
  'ZEEGAP',
  'ZESCAP',
  'ZETAMAX',
  'ZINACEF',
  'XYLOCAINE SPRAY 10%',
  'XYNOSINE',
  'ZEEGAP 75 MG',
  'ZEESPA',
  'ZESTORETIC',
  'ZEXA',
  'ZEZOT',
  'XYNOSINE PAEDS',
  'XOBIX',
  'XOLISAN',
  'XOLISAN 0.1 %',
  'XYLOAID',
  'XYLOAID 10 ML',
  'XCEPT',
  'XIBEN',
  'XIFAXA',
  'XYLOAID WITH ADRENALINE',
  'XYLOCAIN 4%',
  'XATRAL',
  'XAVOR',
  'XELODA',
  'XEPAT EYE',
  'XIVAL',
  'XAVOR DIU',
  'ZOYCIN',
  'ZURIG',
  'ZINCAT OD',
  'ZOLID',
  'ZOLID PLUS',
  'ZOPAN',
  'ZOPAN DS',
  'ZOPENT',
  'ZOVIRAX',
  'ZYLORIC',
  'ZINOROX',
  'ZYNOL',
  'ZINCAT',
]
