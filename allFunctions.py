#from library import *

def calc_mu_hmi(thresh):
        
    try:    
        connectivity=8
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1
        min_size = 100000 
        imagemEmComponentes = np.zeros((output.shape))
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                imagemEmComponentes[output == i + 1] = 1
        
        labels = label(imagemEmComponentes)
        
        props = regionprops(labels)
    
        center_x = props[0]['Centroid'][0]
        center_y = props[0]['Centroid'][1]
        
        EquivDiameter = props[0]['EquivDiameter']
        
        mradius = EquivDiameter/2 
        
        #jx, jy = np.meshgrid(range(1,output.shape[0]+1), range(1,output.shape[1]+1)) #original
        jx, jy = np.meshgrid(range(output.shape[0]), range(output.shape[1]))
        
        jr = np.sqrt(np.power(jx-center_x,2) + np.power(jy-center_y,2))
        
        
        a = 1-np.power(jr/mradius,2)
        a = a.astype('complex')
        
        mu = np.real(np.sqrt(a))
        
        #mu = octave.real(octave.sqrt(a))
        
        ii0 = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        ii1 = np.linspace(1,len(ii0),len(ii0)) #original
        #ii1 = range(1,12)
        
        mu_rings = octave.interp1(ii0, ii1, mu, 'nearest')
        
        #mu_rings = interp1d(ii0, ii1, kind = 'nearest', fill_value = 'extrapolate')(mu)
 
        ''' As 10 linhas seguintes são a opção no caso de não utilizar o interpolador interp1.m do octave ou interp1d do Python.
        mu_rings = np.empty_like(mu)
        mu_rings[:] = np.nan
        
        limites = [1, 0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.075]

        for i in range(len(limites)-1):
            mu_rings[np.where((mu>limites[i+1]) & (mu<=limites[i]))] = ii1[i]
        
        mu_rings[np.where((mu>=0) & (mu<limites[10]))] = ii1[10]
        
        '''
       
        return mu, mu_rings
    
    except:
        print('Erro na função calc_mu_rings')
        return (None, None)


def check_areas():
    
    #partialOutputPath= path + 'partial_output_files\\'

    time = np.loadtxt(partialOutputPath+'time_PY.csv')
    area_c = np.loadtxt(partialOutputPath+'area_c_PY.csv')
    alpha_mu_spot = np.loadtxt(partialOutputPath+'alpha_mu_spot_PY.csv')
    #alpha_mu_spot = np.loadtxt(path+'FileName.txt')
    
    alpha_mu_spot = np.reshape(alpha_mu_spot,(np.size(time),6,11))
    
    #area_c = area_c.reshape([np.size(time),6]) # UMA OPÇÃO AO IF QUE VEM EM SEGUIDA, NECESSÁRIO PARA SOMAR OS VALORES NO CASO DA AREA_C SER UMA MATRIZ UNIDIMENSIONAL
    
    if area_c.ndim > 1:
        count = np.nansum(area_c,1)
    else:
        count = np.nansum(area_c)
    
    mu = np.nanmean(count)
    sigma = np.nanstd(count)
    
    n=count.size
    
    meanMat = matlib.repmat(mu, n, 1)
    sigmaMat = matlib.repmat(sigma, n, 1)
    
    count = count.reshape(meanMat.shape)
    
    outliers = np.abs(count - meanMat) > (3*sigmaMat)
    
    #nOut = np.nansum(outliers) #não usado
    #temp1 = count
    #temp1[np.any(outliers,1),:] = np.nan
    
    area_c[np.any(outliers,1),:] = np.nan
    
    alpha_mu_spot[np.any(outliers,1),:] = np.nan
    
    
    '''QUAL É O OBJETIVO DAS PRÓXIMAS LINHAS (DE 129 À 143) DE CÓDIGO? ESTÁ DANDO ERRO PARA kk == 0 ''' 
    ck, kk = np.unique(time, return_index=True)
    
    '''VERIFICAR AQUI SE OS VALORES RECEBIDOS POR KK SÃO REALMENTE OS PREVISTOS'''
    
    n = area_c.shape[1]
    
    for j in range(n):
        area_c[kk,j] = le_interp(time[kk],area_c[kk,j])
    
    
    n = alpha_mu_spot.shape
    
    for j in range(n[1]):
        for i in range(n[2]):
            alpha_mu_spot[kk,j,i] = le_interp(time[kk],alpha_mu_spot[kk,j,i])
        
    print('\nArquivos salvos em '+partialOutputPath+':\n')
    np.savetxt(partialOutputPath+'check_areas_time_PY.csv', time)
    print('check_areas_time_PY.csv\n')
    
    np.savetxt(partialOutputPath+'check_areas_area_c_PY.csv', area_c)
    print('check_areas_area_c_PY.csv\n')
    
    with open(partialOutputPath+'check_areas_alpha_mu_spot_PY.csv', 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(alpha_mu_spot.shape))
    
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for dataSlice in alpha_mu_spot:
    
            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')

            np.savetxt(outfile, dataSlice, fmt='%-7.5f')
    
    print('check_areas_alpha_mu_spot_PY.csv\n')

    return time, area_c, alpha_mu_spot


def continuumMasks(pathImagemContinua,bw_mask):
    try:
        imagemDaVez = cv2.imread(pathImagemContinua)
    except:
        print("continuumMasks: Imagem " + pathImagemContinua + "não encontrada!")

    try:    
        imagemCinza = cv2.cvtColor(imagemDaVez, cv2.COLOR_BGR2GRAY)
        #ret,thresh = cv2.threshold(imagemCinza,10,1,cv2.THRESH_BINARY)
        
        mu_rings = np.zeros_like(imagemCinza)
        mu_rings[np.where(imagemCinza>10)]=1
        
        ii = np.where(mu_rings>0)
        
        xMedian = np.nanmedian(imagemCinza[ii])
        xStd = np.nanstd(imagemCinza[ii])
        
        h = np.ones((5,5))/25
        I2 = convolve(imagemCinza,h)
        
        x2 = (I2 - xMedian)/xStd
        
        th = -4
        bw = np.zeros_like(x2)
        bw[np.where(x2<=th)] = 1
        
        bw1 = clear_border(bw).astype(int)
        #bw1 = bw.astype(int)

        th = -15
        bw = np.zeros_like(x2)
        bw[np.where(x2<=th)] = 1
        
        '''TESTAR NOVAMENTE A INFORMAÇÃO ABAIX0:
        NO MATLAB, PARA A IMAGEM DE 25/09/2019, 00:00:00 (20190925_000000_Ic_flat_1k.jpg), 
        NÃO "SOBRA" NENHUM TRUE APÓS O IMCLEARBORDER. JÁ NO PYTHON, "SOBRA" 1024 TRUES APÓS O CLEAR_BORDER.'''
        bw2 = clear_border(bw).astype(int) 
        #bw2 = bw.astype(int)
        
        bw_mask[np.where(bw1)] = 6
        bw_mask[np.where(bw2)] = 7
        
        return bw_mask
    
    except:
        print("continuumMasks failed: Erro na identificação e classificação de umbras e penumbras na imagem contínua " + pathImagemContinua) 

def downloadImagens(sufixo, path):
    
    dataDaVez = dataInicial    
    
    previstos=0
    baixados=0
    
    while dataDaVez<=dataFinal:
        
        try:
            #BAIXANDO MAGNETOGRAMAS
            nomeImagem = dataDaVez.strftime('%Y') + dataDaVez.strftime('%m') + dataDaVez.strftime('%d') + '_' + dataDaVez.strftime('%H') + dataDaVez.strftime('%M') + dataDaVez.strftime('%S') + sufixo
            pathImagem = path + '\\' + nomeImagem
        
            if not(os.path.isfile(pathImagem)):
                urlData = url + '/' + dataDaVez.strftime('%Y') + '/' + dataDaVez.strftime('%m') + '/' + dataDaVez.strftime('%d') + '/' + nomeImagem
                imagem = request.urlopen(urlData)
                f = open(pathImagem, 'wb')
                f.write(imagem.read())
                f.close()
                print(pathImagem + '- arquivo baixado!')
            else:
                print(pathImagem + '- arquivo já existente!')
            
            baixados+=1
            # LENDO A IMAGEM:
            #img=mpimg.imread(pathImagem)
            #imgplot = plt.imshow(img)
            #plt.show()
                       
        #except request.URLError:
        #    print(dataDaVez, "não disponível. (Erro de URL)")

        except:
            print("Ocorreu algum erro ao baixar", nomeImagem, ".")
        
        dataDaVez = dataDaVez + timedelta(minutes=resolucao)  
        previstos+=1
    
    '''
    data_atual = date.today()
    print(data_atual.day)
    print(data_atual.month)
    print(data_atual.year)
    print(data_atual.strftime('%d'))
    '''
    print('\nBaixados(ou existentes):'+str(baixados)+'\tPrevistos:'+str(previstos))

def geraAreas():

    qtdDias = abs((dataFinal - dataInicial).days)+1
    print('Processing '+ str(qtdDias) + ' dias...')
    
    dataDaVez = dataInicial
    
    k=0
    
    aux_area = np.empty(6)
    aux_area.fill(np.nan)
    
    area_c = []
    #area_c = np.append(area_c,aux_area)
    
    aux_alpha_mu_spot = np.empty([6,11])
    aux_alpha_mu_spot.fill(np.nan)
    
    alpha_mu_spot = []
    #alpha_mu_spot = np.append(alpha_mu_spot,aux_alpha_mu_spot)
    
    time = []
    
    while dataDaVez<=dataFinal:
        
        ano = dataDaVez.strftime('%Y')
        mes = dataDaVez.strftime('%m')
        dia = dataDaVez.strftime('%d')
        horas = dataDaVez.strftime('%H')
        minutos = dataDaVez.strftime('%M')
        segundos = dataDaVez.strftime('%S')
    
        if ((horas=='00') | (horas == '06') | (horas == '12') | (horas == '18')) & ((minutos == '00') | (minutos == '30')):

            print('\nProcessando dia '+str(dataDaVez))
    
            nomeImagem = ano + mes + dia + '_' + horas + minutos + segundos
            
            try:
                tHours = dataDaVez.hour/24
                tMinutes = dataDaVez.minute/(24*60)
                tSeconds = dataDaVez.second/(24*60*60)
                t_obs_preliminary = dataDaVez.toordinal()+tHours+tMinutes+tSeconds
                #t_obs_preliminary = dataDaVez.toordinal()+tHours+tMinutes+tSeconds+366
            
                area_disk, bw_mask = imageMasks(nomeImagem)
                
                if k == 0:
                    a = np.zeros_like(bw_mask)
                    
                    a[np.where(bw_mask>0)] = 1
                    a = np.uint8(a)                
                    
                    mu, mu_rings = calc_mu_hmi(a)
                    
                    #ret,thresh = cv2.threshold(bw_mask,0,1,cv2.THRESH_BINARY)
                    #thresh = np.uint8(thresh)
                    #mu, mu_rings = calc_mu_hmi(thresh)
                    
         #           area_c = area_c.reshape(1,6)
         #           alpha_mu_spot = alpha_mu_spot.reshape(1,6,11)
                ndisk = np.count_nonzero(mu_rings)                
                
                for i in range(2,8):
                    if i==5:
                        bw1 = (bw_mask == 5) | (bw_mask == 6) | (bw_mask == 7)
                        itemp = np.where(bw1)
                    elif i == 6:
                        bw1 = bw_mask == 6
                        itemp = np.where(bw1)
                    elif i == 7:
                        bw1 = bw_mask == 7
                        itemp = np.where(bw1)
                    else:
                        bw1 = bw_mask == i
                        itemp = np.where(bw1)
                    
                    itemp = np.asarray(itemp)
                    
                    if itemp.shape[1]>0:
                        aux_area[i-2] = itemp.shape[1]/area_disk
                    else:
                        aux_area[i-2] = 0
                    
                    '''VALORES DE MU_RINGS APRESENTAM DIFERENÇAS ENTRE MATLAB E PYTHON'''
                    for m in range(1,12):
                        temp = bw1*(mu_rings==m)
                        alpha_mu_preliminary = np.nansum(temp) / ndisk
                        aux_alpha_mu_spot[i-2][m-1] = alpha_mu_preliminary
                
                area_c = np.append(area_c, [aux_area])
                
                alpha_mu_spot = np.append(alpha_mu_spot, [aux_alpha_mu_spot])
                        
                time = np.append(time,t_obs_preliminary)
                
                aux_area = np.empty(6)
                aux_area.fill(np.nan)
                
                aux_alpha_mu_spot = np.empty([6,11])
                aux_alpha_mu_spot.fill(np.nan)
                
                k = k+1
            
                area_c = area_c.reshape(k,6)
                alpha_mu_spot = alpha_mu_spot.reshape(k,6,11)

            except:
                print('Masks not defined for ' + str(nomeImagem))    
            
        dataDaVez = dataDaVez + timedelta(minutes=resolucao)
    
    print('\nArquivos salvos em '+partialOutputPath+' :\n')
    np.savetxt(partialOutputPath+'\\time_PY.csv', time)
    print('time_PY.csv\n')
    
    np.savetxt(partialOutputPath+'\\area_c_PY.csv', area_c)
    print('area_c_PY.csv\n')
    
    with open(partialOutputPath+'\\alpha_mu_spot_PY.csv', 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(alpha_mu_spot.shape))
    
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for dataSlice in alpha_mu_spot:
            outfile.write('# New slice\n')
            np.savetxt(outfile, dataSlice, fmt='%-7.5f')
    
    print('alpha_mu_spot_PY.csv\n')
    #return alpha_mu_spot,area_c,time

def imageMasks(nomeImagem):

    #nomeImagemContinua = nomeImagem + sufixoContinua
    #nomeImagemMag = nomeImagem + sufixoMag
    
    pathImagemContinua = pathContinua + '\\' + nomeImagem + sufixoContinua
    pathImagemMag = pathMag + '\\' + nomeImagem + sufixoMag            
    
    bw_mask = np.zeros((1024,1024))
    
    try:
        bw_mask, area_disk = magMasks(pathImagemMag,bw_mask)
    except:
        print('magMasks.py failed to ' + str(nomeImagem))
        
    try:
        bw_mask = continuumMasks(pathImagemContinua,bw_mask)
        #print(bw_mask)
    except:
        print('continuumMasks.py failed to ' + str(nomeImagem))
  
    return area_disk, bw_mask            

def le_interp(t,x):

    y = x;

    ii = np.squeeze(np.where(~np.isnan(x)))
    ii1 = np.squeeze(np.where(np.isnan(x)))
    
    y[ii1] = interp1d(t[ii],x[ii], fill_value = 'extrapolate')(t[ii1])
    
    return y


def magMasks(pathImagemMag,bw_mask):
    
    try:
        imagemDaVez = cv2.imread(pathImagemMag)
        try:    
            imagemCinza = cv2.cvtColor(imagemDaVez, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(imagemCinza,10,1,cv2.THRESH_BINARY)
            
            '''MU_RINGS ESTÁ VINDO SEM NAN, DIFERENTE DO QUE ACONTECE NO MATLAB, O QUE ESTÁ DESTOANDO OS VALORES CALCULADOS PARA O BW_MASK, EM RELAÇÃO AOS CALCULADOS NO MATLAB'''
            mu, mu_rings = calc_mu_hmi(thresh)
            
            area_disk = np.count_nonzero(mu>0)
    
        except:
            print("magMasks: Erro nas etapas de limpeza e de definição do disco solar e dos anéis" + pathImagemMag)  
        
        try:
            bw1 = imagemCinza>=(128+20)
            bw2 = imagemCinza<=(128-20)
            bw3 = bw1 | bw2
            bw4 = np.uint8(bw3 & (mu_rings > 0))
     
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(bw4, connectivity=4)
            sizes = stats[1:, -1]
            nb_components = nb_components - 1
            min_size = 10 
            bw5 = np.zeros((output.shape))
    
            for i in range(0, nb_components):
                if sizes[i] >= min_size:
                    bw5[output == i + 1] = 1
            
            labels = label(bw5, connectivity=1)
            
            props = regionprops(labels)
            
            area = []
    
            for region in props:
                area.append(1e6*region.area/area_disk)
            #area_total.append(area) #apenas uma verificação das áreas totais calculadas
    
            '''VERIFICAR SE NA LINHA ABAIXO DEVE REALMENTE SER EXCLUÍDA A PRIMEIRA ÁREA (VALOR MÁXIMO), CUJO VALOR NÃO CONDIZ COM OS RESULTADOS EM MATLAB (em alguns casos!!!)'''
            #area = np.asarray(area[1:])
            #props = props[1:]
            area = np.asarray(area)
        except:
            print("magMasks: Erro na etapa de pré classificação da imagem " + pathImagemMag)  
            
        try:  
            '''ESSES VALORES DE THRESHOLDS DEVEM FICAR EM UM ARQUIVO MESMO?? COMO SÃO ESTIMADOS???'''
            th1_m = float(16.6883)
            th2_m = float(24.8066)
            th3_m = float(89.0673)
            
            bw_small = np.zeros(bw5.shape)
            ii = np.where(area <= th1_m)
            ii = np.array(ii)[0]
            flag_small = 0        
            if ii.size > 0:
                flag_small = 1
                for i in range (ii.size):
                    x = props[ii[i]]['coords'][:,0]
                    y = props[ii[i]]['coords'][:,1]
                    bw_small[x,y] = 1 
                    
            bw_media1 = np.zeros(bw5.shape)            
            ii = np.where((area > th1_m) & (area <= th2_m))
            ii = np.array(ii)[0]
            flag_media1 = 0        
            if ii.size > 0:
                flag_media1 = 1
                for i in range (ii.size):
                    x = props[ii[i]]['coords'][:,0]
                    y = props[ii[i]]['coords'][:,1]
                    bw_media1[x,y] = 1 
            
            bw_media2 = np.zeros(bw5.shape)            
            ii = np.where((area > th2_m) & (area <= th3_m))
            ii = np.array(ii)[0]
            flag_media2 = 0        
            if ii.size > 0:
                flag_media2 = 1
                for i in range (ii.size):
                    x = props[ii[i]]['coords'][:,0]
                    y = props[ii[i]]['coords'][:,1]
                    bw_media2[x,y] = 1 
    
            bw_large = np.zeros(bw5.shape)            
            ii = np.where(area > th3_m)
            ii = np.array(ii)[0]
            flag_large = 0        
            if ii.size > 0:
                flag_large = 1
                for i in range (ii.size):
                    x = props[ii[i]]['coords'][:,0]
                    y = props[ii[i]]['coords'][:,1]
                    bw_large[x,y] = 1 
    
    
            
            bw_mask[np.where(mu_rings > 0)] = 1               
            
            if flag_small:
                bw_mask[np.where(bw_small==1)] = 2
            if flag_media1:
                bw_mask[np.where(bw_media1==1)] = 3
            if flag_media2:
                bw_mask[np.where(bw_media2==1)] = 4
            if flag_large:
                bw_mask[np.where(bw_large==1)] = 5
    
            return bw_mask, area_disk
        
        except:
            print("magMasks: Erro no processo de classificação da imagem " + pathImagemMag)
        
    except:
        print("magMasks failed: Imagem " + pathImagemMag + "não encontrada!")


def model_mdi_02_03():
    
    lver = 2
    lsubver = 3
    ierror = 0
    
    time, area_c, alpha_mu_spot = check_areas()
    
    time_tim, tsi_tim, tsi_tim_sig = read_tim_tsi()
    
    dt = 1/4
    
    tHours = dataInicial.hour/24
    tMinutes = dataInicial.minute/(24*60)
    tSeconds = dataInicial.second/(24*60*60)
    #dI = dataInicial.toordinal()+366+tHours+tMinutes+tSeconds
    dI = dataInicial.toordinal()+tHours+tMinutes+tSeconds
    
    tHours = dataFinal.hour/24
    tMinutes = dataFinal.minute/(24*60)
    tSeconds = dataFinal.second/(24*60*60)
    #dF = dataFinal.toordinal()+366+tHours+tMinutes+tSeconds
    dF = dataFinal.toordinal()+tHours+tMinutes+tSeconds
    
    #dI = date.toordinal(dataInicial)+366
    #dF = date.toordinal(dataFinal)+366+1/4
    period = np.arange(dI,dF,dt)
    
    t = []
    tsi_tim_t = []
    alpha = np.zeros((11,6,len(period)))
    
    for j in range(len(period)):
        t1 = period[j]
        
        jj = np.where((time >= (t1-dt/2)) & (time < (t1+dt/2))) 
        
        for i in range(11):
            for k in range(6):
                temp = alpha_mu_spot[jj,k,i] #VERIFICAR SE OS ÍNDICES BATEM COM A ORDEM GRAVADA NO CHECK_AREAS()
                kk = np.where(np.isfinite(temp))
                alpha[i,k,j] = np.mean(temp[kk])
        
        jj = np.where((time_tim >= (t1 - dt/2)) & (time_tim < (t1 + dt/2)))
        
        if (np.squeeze(jj)).size > 0:
            tsi_tim_t.append(tsi_tim[jj])
        else:
            tsi_tim_t.append(np.nan)
        
        t.append(t1)
        
    t = np.array(t)
    tsi_tim_t = np.array(tsi_tim_t)
    
    for i in range(11):
        for k in range(6):
            alpha[i,k,0] = alpha[i,k,1]
    
    for i in range(11):
        for k in range(6):
            alpha[i,k,:] = le_interp(t, np.squeeze(alpha[i,k,:]))
    
    '''???'''
    F_f = np.squeeze(alpha[:,3,:] - alpha[:,4,:] - alpha[:,5,:])
    
    '''QUE DATA É ESSA??? '''
    kk = np.array(np.where((t >= date.toordinal(dataInicial)) & np.isfinite(F_f[0,:])))
    kk = np.squeeze(kk)
    ds = 1
    
    time0 = t[kk[ds:]]
    T = tsi_tim_t[kk[ds:]]
      
    P = np.array([np.squeeze(alpha[:10,2,kk[:len(kk)-(ds)]]), 
                  np.squeeze(alpha[:10,3,kk[:len(kk)-(ds)]]), 
                  np.squeeze(alpha[:10,4,kk[:len(kk)-(ds)]]),
                  np.squeeze(alpha[:10,5,kk[:len(kk)-(ds)]])])

    with open(partialOutputPath+'\\P_PY3D.csv', 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(P.shape))

        slice=0
        #print('Saving slices: '),
        for dataSlice in P:
            outfile.write('# New slice\n')
            np.savetxt(outfile, dataSlice, fmt='%-7.5f')
            slice+=1
            #print(str(slice)+'\t'),
        
    print('Arquivo salvo: '+partialOutputPath+'\\P_PY3D.csv')
    
    P = P.reshape(P.shape[0]*P.shape[1],len(period)-1)
    T = T.transpose()
    
    #nans = np.isnan(T)
    #x = lambda z: z.nonzero()[0]
    #T[nans] = np.interp(x(nans), x(~nans), T[~nans])

    #P = np.transpose(P)
    print('\nMatrizes não normalizadas salvas em '+partialOutputPath+' :\n')
    np.savetxt(partialOutputPath+'\\P_PY.csv', P)
    print('P_PY.csv\n')
    
    np.savetxt(partialOutputPath+'\\T_PY.csv', T)
    print('T_PY.csv\n')
    
    #rnn(P,T)

def read_tim_tsi():
    
    #path='C:\\Users\Ami\Dropbox\Tese\Traduzido\\'
    tsiFile = 'sorce_tsi_L3_c06h_latest.txt'
    
    #url = 'http://lasp.colorado.edu/data/sorce/tsi_data/six_hourly/' + tsiFile
   
    filePath = path+tsiFile
    
    #uploadedFile = request.urlopen(url)

    #f = open(filePath, 'wb')

    #f.write(uploadedFile.read())
    #f.close()

    

    with open(filePath,'r') as f: 
        #read only data, ignore headers 
        lines = f.readlines()[134:] 
        # create the arrays 
        data = ''
        time_tim = np.zeros(len(lines))
        
        #nominal_date_jdn = np.zeros(len(lines))
        #avg_measurement_date_jdn = np.zeros(len(lines))
        #std_dev_measurement_date = np.zeros(len(lines))
        tsi_tim = np.zeros(len(lines))
        #temp1 = np.zeros(len(lines))
        #temp2 = np.zeros(len(lines))
        #temp3 = np.zeros(len(lines))
        tsi_tim_sig = np.zeros(len(lines))
        
        # convert strings to floats and put into arrays 
        for i in range(len(lines)):
            data, data2, data3, data3, tsi_tim[i], data4, data5, data6, tsi_tim_sig[i], seinao, seinao, seinao, seinao, seinao, seinao= lines[i].split()
            
            yyyy = int(data[0:4])
            mm = int(data[4:6])
            dd = float(data[6:])
            hh = dd - int(dd)
            dd = int(dd)
            time_tim[i] = date.toordinal(datetime(yyyy,mm,dd))+hh
            
            tsi_tim[i] = float(tsi_tim[i])
            tsi_tim_sig[i] = float(tsi_tim_sig[i])
    
    tsi_tim[np.where(tsi_tim == 0)] = np.nan
    tsi_tim_sig[np.where(tsi_tim_sig == 0)] = np.nan
    #measurement_uncertainty_1au(measurement_uncertainty_1au==0)=nan;
    
    n1 = len(time_tim)
    n2 = len(tsi_tim)
    
    if np.not_equal(n1,n2):
        n = min(n1,n2)
        tsi_tim = tsi_tim[:n]
        time_tim = time_tim[:n]
        tsi_tim_sig = tsi_tim_sig[:n]
    
    return time_tim, tsi_tim, tsi_tim_sig


def rnn(P,T):
    np.random.seed(7)

    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    
    pn = scaler.fit_transform(np.transpose(P))
    tn = scaler.fit_transform(T)
    #pn = np.transpose((P - np.nanmean(P, axis=0))/np.nanstd(P, axis=0))
    #tn = (T - np.nanmean(T))/np.nanstd(T)
    
    seriesSize = len(tn)
    print('\nArquivos salvos em '+path+' :\n')
    np.savetxt(path+'\\pn_PY.csv', pn)
    print('pn_PY.csv\n')
    
    np.savetxt(path+'\\tn_PY.csv', tn)
    print('tn_PY.csv\n')

    trainSize = int(seriesSize*0.8)
    testSize = seriesSize - trainSize
    #xTrain, xTest, yTrain, yTest = train_test_split(pn, np.squeeze(tn), test_size=0.2, shuffle=False)
    
    xTrain, xTest = pn[:trainSize,:], pn[trainSize:seriesSize,:] 
    yTrain, yTest = tn[:trainSize], tn[trainSize:seriesSize]
    
    xTrain = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1]))
    xTest = xTest.reshape((xTest.shape[0], 1, xTest.shape[1]))
    #pn = np.transpose(pn).reshape((pn.shape[0], 1, pn.shape[1]))
    
    model = Sequential()
#    model.add(SimpleRNN(units=10, input_shape=(xTrain.shape[1],xTrain.shape[2]), activation='sigmoid'))
    model.add(SimpleRNN(units=10, input_shape=(xTrain.shape[1],xTrain.shape[2]), activation='relu'))
    model.add(Dropout(0.3))
    #model.add(GRU(units=1, input_shape=(xTrain.shape[0],xTrain.shape[1],xTrain.shape[2]), return_sequences=True))
    #model.add(GRU(units=2, input_shape=(xTrain.shape[1],xTrain.shape[2])))
    model.add(Dense(1, activation='linear'))
    #model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    
    #f = model.fit(xTrain, yTrain, epochs=2000, batch_size=xTrain.shape[2], validation_data=(xTest,yTest), shuffle=False)
    callback = []
    callback.append(Callback.EarlyStopping(monitor='val_loss', patience=5))
    callback.append(Callback.ModelCheckpoint(filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5',monitor='val_loss'))

    f = model.fit(xTrain, yTrain, epochs=500, batch_size = 30, validation_data=(xTest,yTest), shuffle=False, callbacks=callback)
    #model.fit(pn, np.squeeze(tn), epochs=1000, validation_split=0.05) # batch_size=1, verbose=2)
    # make predictions
    
    plt.figure()
    plt.plot(f.history['loss'], label='treinamento')
    plt.plot(f.history['val_loss'], label='teste')
    plt.title('Erro de Treinamento e Teste', loc='left', fontsize=16)
    plt.legend()

    trainPredictedY = model.predict(xTrain)
    testPredictedY = model.predict(xTest)    
    
    plt.figure()
    plt.scatter(testPredictedY,yTest)
   # plt.plot([min(testPredictedY),max(testPredictedY)],[min(yTest),max(yTest)])
    plt.plot([-2,2],[-2,2])
    #plt.plot(tn[0,:],'b*-')

    plt.figure()
    plt.scatter(testPredictedY,yTest)
   # plt.plot([min(testPredictedY),max(testPredictedY)],[min(yTest),max(yTest)])
    plt.plot([-2,2],[-2,2])

    plt.figure()
    plt.plot(yTrain,'b*-',label='desejado')
    plt.plot(trainPredictedY,'g*-',label='obtido')
    plt.title('Desempenho Treinamento', loc='left', fontsize=16)
    plt.legend()

    #plt.plot(yTrain,yTrain,'k-',yTrain,trainPredictedY,'b*')
    plt.ylim((-2,2))

    
    plt.figure()
    plt.plot(yTest,'b*-',label='desejado')
    plt.plot(testPredictedY,'g*-',label='obtido')
    plt.ylim((-2,2))
    plt.title('Desempenho Teste', loc='left', fontsize=16)
    plt.legend()

    #plt.plot(yTest,yTest,'k-',yTest,testPredictedY,'b*')
    #plt.ylim((-2,2))
    
    '''AS LINHAS SEGUINTES invertem a normalização para a escala original'''
    #trainPredict = scaler.inverse_transform(trainPredict)
    #yTrain = scaler.inverse_transform([yTrain])
    #testPredict = scaler.inverse_transform(testPredict)
    #yTest = scaler.inverse_transform([yTest])
    
    # calculate root mean squared error
    #trainScore = math.sqrt(mean_squared_error(yTrain[0], trainPredict[:,0]))
    #trainScore = math.sqrt(mean_squared_error(yTrain, trainPredict[:,0]))
    #print('Train Score: %.2f RMSE' % (trainScore))
    #testScore = math.sqrt(mean_squared_error(yTest[0], testPredict[:,0]))
    #print('Test Score: %.2f RMSE' % (testScore))
    
    

''' 
    A função imagesMasks quando ainda estava com o atual geraAreas.
def imagesMasks(path, dataInicial, dataFinal, resolucao):
    
    sufixoContinua = '_Ic_flat_1k.jpg'
    sufixoMag = '_M_1k.jpg'
    
    dataInicial = datetime.strptime(dataInicial,'%d/%m/%Y %H:%M:%S')
    dataFinal = datetime.strptime(dataFinal,'%d/%m/%Y %H:%M:%S')
    
    #qtdDias = abs((dataFinal - dataInicial).days)+1
    #print('Quantidade de dias: ', qtdDias)
    
    dataDaVez = dataInicial
    
    #path = 'C:\\Users\\Ami\\Dropbox\\Tese\\Traduzido\\'
    pathMag = path + 'mag'
    pathContinua = path + 'continuum'
    
    k=0
    
    aux_area = np.empty(6)
    aux_area.fill(np.nan)
    
    area_c = []
    #area_c = np.append(area_c,aux_area)
    
    aux_alpha_mu_spot = np.empty([6,11])
    aux_alpha_mu_spot.fill(np.nan)
    
    alpha_mu_spot = []
    #alpha_mu_spot = np.append(alpha_mu_spot,aux_alpha_mu_spot)
    
    area_disk_c = []
    time = []
    
    while dataDaVez<=dataFinal:
        
        ano = dataDaVez.strftime('%Y')
        mes = dataDaVez.strftime('%m')
        dia = dataDaVez.strftime('%d')
        horas = dataDaVez.strftime('%H')
        minutos = dataDaVez.strftime('%M')
        segundos = dataDaVez.strftime('%S')

        if ((horas=='00') | (horas == '06') | (horas == '12') | (horas == '18')) & ((minutos == '00') | (minutos == '30')):
            print('\nProcessando dia '+str(dataDaVez))
            
            nomeImagem = ano + mes + dia + '_' + horas + minutos + segundos
            
            nomeImagemContinua = nomeImagem + sufixoContinua
            nomeImagemMag = nomeImagem + sufixoMag
            
            pathImagemContinua = pathContinua + '\\' + nomeImagemContinua
            pathImagemMag = pathMag + '\\' + nomeImagemMag
            
            tHours = dataDaVez.hour/24
            tMinutes = dataDaVez.minute/(24*60)
            tSeconds = dataDaVez.second/(24*60*60)
            t_obs_preliminary = dataDaVez.toordinal()+366+tHours+tMinutes+tSeconds
            
            bw_mask = np.zeros((1024,1024))
        
            try:
                bw_mask, area_disk = magMasks(pathImagemMag,bw_mask)
            except:
                print('magMasks.py failed to ' + str(dataDaVez))
                
            try:
                bw_mask = continuumMasks(pathImagemContinua,bw_mask)
                #print(bw_mask)
            except:
                print('continuumMasks.py failed to ' + str(dataDaVez))
                
               
            #np.savetxt(path + nomeImagem + '_area.txt', bw_mask)
            area_disk_c = np.append(area_disk_c,area_disk)
            #bw_msk = bw_mask.astype('int')
        
        ##if ((horas=='00') | (horas == '06') | (horas == '12') | (horas == '18')) & ((minutos == '00') | (minutos == '30')):
            
            if k == 0:
                a = np.zeros_like(bw_mask)
                
                a[np.where(bw_mask>0)] = 1
                a = np.uint8(a)                
                
                mu, mu_rings = calc_mu_hmi(a)
                
                #ret,thresh = cv2.threshold(bw_mask,0,1,cv2.THRESH_BINARY)
                #thresh = np.uint8(thresh)
                #mu, mu_rings = calc_mu_hmi(thresh)
                
     #           area_c = area_c.reshape(1,6)
     #           alpha_mu_spot = alpha_mu_spot.reshape(1,6,11)
            ndisk = np.count_nonzero(mu_rings)                
            
            for i in range(2,8):
                if i==5:
                    bw1 = (bw_mask == 5) | (bw_mask == 6) | (bw_mask == 7)
                    itemp = np.where(bw1)
                elif i == 6:
                    bw1 = bw_mask == 6
                    itemp = np.where(bw1)
                elif i == 7:
                    bw1 = bw_mask == 7
                    itemp = np.where(bw1)
                else:
                    bw1 = bw_mask == i
                    itemp = np.where(bw1)
                
                itemp = np.asarray(itemp)
                
                if itemp.shape[1]>0:
                    aux_area[i-2] = itemp.shape[1]/area_disk_c[k]
                else:
                    aux_area[i-2] = 0
                
                #VALORES DE MU_RINGS APRESENTAM DIFERENÇAS ENTRE MATLAB E PYTHON
                for m in range(1,12):
                    temp = bw1*(mu_rings==m)
                    alpha_mu_preliminary = np.nansum(temp) / ndisk
                    aux_alpha_mu_spot[i-2][m-1] = alpha_mu_preliminary
            
            area_c = np.append(area_c, [aux_area])
            
            alpha_mu_spot = np.append(alpha_mu_spot, [aux_alpha_mu_spot])
                    
            time = np.append(time,t_obs_preliminary)
            
            aux_area = np.empty(6)
            aux_area.fill(np.nan)
            
            aux_alpha_mu_spot = np.empty([6,11])
            aux_alpha_mu_spot.fill(np.nan)
            
            k = k+1
        
            area_c = area_c.reshape(k,6)
            alpha_mu_spot = alpha_mu_spot.reshape(k,6,11)
        
        dataDaVez = dataDaVez + timedelta(minutes=resolucao)
    
    print('\nArquivos salvos em '+path+' :\n')
    np.savetxt(path+'\\time.csv', time)
    print('time.csv\n')
    
    np.savetxt(path+'\\area_c.csv', area_c)
    print('area_c.csv\n')
    
    with open(path+'\\alpha_mu_spot.csv', 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(alpha_mu_spot.shape))
    
        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for dataSlice in alpha_mu_spot:
    
            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')

            np.savetxt(outfile, dataSlice, fmt='%-7.5f')
    
    
    print('alpha_mu_spot.csv\n')
    return alpha_mu_spot,area_c,time
    '''