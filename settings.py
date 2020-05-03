path = 'D:\\Amita\\Doutorado\\Tese\\Traduzido\\'
partialOutputPath = path + 'partial_output_files\\'
outputPath = path + 'output\\'
#dataInicial = '07/03/2012 00:00:00'
#dataFinal = '09/03/2012 23:30:00'

#dataInicial = '15/03/2012 00:00:00'
#dataFinal = '30/03/2012 23:30:00'

dataInicial = '05/11/2011 00:00:00'
dataFinal = '30/03/2012 23:45:00'

dataInicial = datetime.strptime(dataInicial,'%d/%m/%Y %H:%M:%S')
dataFinal = datetime.strptime(dataFinal,'%d/%m/%Y %H:%M:%S')

resolucao = 30
#resolucao = 60*6

url = "http://jsoc.stanford.edu/data/hmi/images"

sufixoContinua = '_Ic_flat_1k.jpg'
sufixoMag = '_M_1k.jpg'

#sufixoContinua = '_Ic_flat_256.jpg'
#sufixoMag = '_M_256.jpg'

pathMag = path + 'mag'
pathContinua = path + 'continuum'       

#pathMag = path + 'mag teste cnn'
#pathContinua = path + 'continuum teste cnn'       