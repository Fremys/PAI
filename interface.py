
rtgtrgtrgtrgrtggrtgrtrtgrtrtgrtgrtg
from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image, ImageEnhance
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import time

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image

#Definindo classe para modificação da imagem
class processImage:
    
    def __init__(self, fileName, root):
        self.root = root
        
        self.filePath = fileName
        self.img = Image.open(fileName)
        self.img_c = Image.open(fileName)
        self.img_cut = Image.open(fileName)
        self.img_cut_c = Image.open(fileName)
        
        self.currentResult = ""
        self.time_execution_current = 0
        
        self.label = None
        self.label_time_execution = None
        
        self.slide_c = None
        self.slide_z = None
        
        self.label_txt1 = None
        self.canvas_txt2 = None
        self.labelResult = None
        
        self.buttonSF = None
        self.buttonCN = None
        self.buttonCB = None
        self.buttonSegment = None
        
        self.x_zoom = 200
        self.y_zoom = 200
        
        self.contrast_level = 1
        
    def show(self):
        
        #reestruturando imagem
        new_img = self.trow_img()

        self.root.title("Process Image")
        # self.root.geometry("1280x720")
        
        #definir tela frame principal de exibição
        mainFrame = ttk.Frame(self.root, padding=("10 10 10 10"))
        mainFrame.grid(column=0, row=0, sticky=(N, W, E, S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        #definir botão para seleção do arquivo
        self.buttonSF = Button(mainFrame, text="imagem", borderwidth=2, command=self.selected_file) 
        self.buttonSF.grid(column=2, row=5)
        
        #definir botão para execução da predição binária
        self.buttonCB = Button(mainFrame, text="Classificação B",  borderwidth=2, command=self.detectedBinary )
        self.buttonCB.grid(column=3, row=5)
        
        #definir botão para execução da predição não binária
        self.buttonCB = Button(mainFrame, text="Classificação",  borderwidth=2, command=self.detectedNoBinnary )
        self.buttonCB.grid(column=3, row=6)
        
        #definir componente para exibição da imagem
        self.label = Label(mainFrame, image=new_img, borderwidth=2, relief="sunken")
        self.label.bind("<Button 1>", self.define_zoom)
        self.label.grid(column=2, row=2)
        
        #Definir descrição do contraste
        self.label_txt1 = Label(mainFrame, text="Contraste", font="Courier", height=2)
        self.label_txt1.grid(column=2,  row=3)
        
        #definir slide para contraste
        self.slide_c = Scale(mainFrame, width=15, from_=0, to=100, orient="horizontal", command=self.contrast)
        self.slide_c.grid(column=2,  row=4)
        
        # Segmentar a imagem
        self.buttonSegment = Button(mainFrame, text="Segmentar",  borderwidth=2, command=self.segment_mammary_region)
        self.buttonSegment.grid(column=1, row=4)
        
        #Definir descrição do zoom
        self.canvas_txt2 = Canvas(mainFrame, width=50, height=150)
        self.canvas_txt2.create_text(20, 80,text="Zoom", angle=90)
        self.canvas_txt2.grid(column=3, row=2)

        #Definir slide zoom
        self.slide_z = Scale(mainFrame, width=15, from_=1, to=100, orient="vertical", command=self.zoom)
        self.slide_z.grid(column=4, row=2)
        
        #Definir label para resultado
        self.labelResult= Label(mainFrame, text="RESULTADO", font="Courier", height=4)
        self.labelResult.grid(column=2, row=0)

        #Definir label para resultado
        self.labelResult= Label(mainFrame, text=self.currentResult, font="Courier", height=2)
        self.labelResult.grid(column=2, row=1)
        
        #Definir label para tempo de execução
        self.label_time_execution = Label(mainFrame, text="Tempo: " + str(self.time_execution_current), font="Courier", height=3)
        self.label_time_execution.grid(column=1, row=4)
        
        self.root.mainloop()
        
    def trow_img(self):
        # redimensionar imagens
        img_r = self.img.resize((400,400))
        
        self.img = img_r
        self.img_c = img_r
        self.img_cut = img_r
        self.img_cut_c = img_r
        
        #converter imagem principal
        new_img = ImageTk.PhotoImage(img_r)
        
        return new_img
    
    def contrast(self, value):
        # nomrmalizar valor do zoom
        new_value = 1 if value == "0" else int(value)
        # salvar ultimo valor de constraste
        self.contrast_level = new_value
        
        # verificar se o constraste é outro
        if(new_value != 0):
            #  aplicar constraste
            img_m_c = ImageEnhance.Contrast(self.img_cut).enhance(int(new_value))
            img_m_nc = ImageEnhance.Contrast(self.img).enhance(int(new_value))
            
            #salavar imagem com constraste para aplicação
            self.img_c = img_m_nc
            self.img_cut_c = img_m_c
            
            #converter imagem para mostrar na tela
            new_img = ImageTk.PhotoImage(img_m_c)
            
            #mostrar na tela
            self.label.config(image=new_img)
            self.label.image = new_img
                
    def selected_file(self):
        # Abrir o diálogo de seleção de arquivo
        file_path = askopenfilename()
        
        if file_path:
            # Salvar caminho da imagem
            self.filePath = file_path
            
            #Abrir imagem e salvá-la globalmente
            self.img = Image.open(file_path)
            
            #Abrir imagem na interface gráfica
            new_img = self.trow_img()
            self.label.config(image=new_img)
            self.label.image = new_img
    
    def detectedBinary(self):
        self.classDetected(True)
    
    def detectedNoBinnary(self):
        self.classDetected(False)
    
    def classDetected(self, b):
        # Definir possíveis classes names
        classPredict = ["D", "E", "F", "G"]
        classBinaryPredict = ["I","II"]

        self.img_cut_c.save('nova_imagem.png')
        
       # Carrega a imagem e redimensiona
        img = image.load_img('nova_imagem.png', target_size=(256, 256, 3))

        # Normaliza a imagem
        img_array = image.img_to_array(img)
        img_array /= 255.

        # Cria um objeto EagerTensor a partir da imagem normalizada
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Adiciona uma dimensão extra para o modelo
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        
        #marcar tempo de inicio
        init_time = time.time()
        
        # Verificar se a classificação é binária ou não
        if(b):
            # Carrega o modelo
            model = load_model('ResNet50Binary.h5')

            # Faz a predição na imagem
            predictions = model.predict(img_tensor)

            # Converte a saída para um rótulo de texto
            # SELECIONAR A CLASSE - BINARIA OU COM 4 CLASSES (classe,classeBinary)
            predicted_label = classBinaryPredict[np.argmax(predictions)]

            # Exibe a classe prevista
            self.currentResult = predicted_label
            
            self.labelResult.config(text=self.currentResult)
        else:
            # Carrega o modelo
            model = load_model('ResNet50.h5')
            
            # Faz a predição na imagem
            predictions = model.predict(img_tensor)

            # Converte a saída para um rótulo de texto
            # SELECIONAR A CLASSE - BINARIA OU COM 4 CLASSES (classe,classeBinary)
            predicted_label = classPredict[np.argmax(predictions)]
            
            # Exibe a classe prevista
            self.currentResult = predicted_label
            
            self.labelResult.config(text=self.currentResult)
            
        #marcar tempo apos execução
        time_end = time.time()
        #salvar tempo de execução
        self.time_execution_current_time = time_end - init_time
        self.label_time_execution.config(text="Tempo: " + str(self.time_execution_current_time))
            
    def define_zoom(self, event):
        self.x_zoom = event.x 
        self.y_zoom = event.y
    
    def zoom(self, value):
        
        # normalizar o valor do zoom
        value_convert = (int(value)/100) * 200
        
        # dar zoom na foto
        calc_zoom = 200 - int(value_convert)
        
        # verificar se o zoom esta passando dos limites da tela e recalcular
        up = self.y_zoom-calc_zoom
        down = self.y_zoom+calc_zoom
        right = self.x_zoom+calc_zoom
        left = self.x_zoom-calc_zoom
        
        # recalcular
        if(left <= 0):
            left = 0
            right += calc_zoom
        else:
            if(right >= 400):
                right = 400
                left -= calc_zoom
                
        if(up < 0):
            print("teste")
            up = 0
            down += calc_zoom
        else:
            if(down >= 400):
                down = 400
                up -= calc_zoom
        
        # dar o zoom fazendo o corte na imagem
        img_cut = self.img_c.crop((0 if left < 0 else left, 0 if up < 0 else up, 400 if right > 400 else right, 400 if down > 400 else down))
        #redmiensionar para ampliação da area do zoom
        img_z = img_cut.resize((400,400), Image.ANTIALIAS)
        #slavar imagem cortada para aplicação do contraste
        self.img_cut = img_z
        self.img_cut_c = img_z
        
        # converter imagem para mostra-la na tela
        new_img = ImageTk.PhotoImage(img_z)
        
        #mostrar imagem na tela
        self.label.config(image=new_img)
        self.label.image = new_img
    
    def segment_mammary_region(self):
        image = self.img_cut_c
        
        # Converter a imagem para escala de cinza
        image_gray = image.convert("L")

        # Converter a imagem para um array NumPy
        image_array = np.array(image_gray)

        # Aplicar limiarização adaptativa para segmentar a região da mama
        _, thresholded = cv2.threshold(
            image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Encontrar os contornos dos objetos na imagem binarizada
        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Encontrar o contorno da mama (maior área)
        largest_contour = max(contours, key=cv2.contourArea)

        # Criar uma máscara em branco do tamanho da imagem
        mask = np.zeros_like(image_array)

        # Desenhar o contorno da mama na máscara
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

        # Aplicar a máscara na imagem original usando a função bitwise_and
        result_array = cv2.bitwise_and(image_array, image_array, mask=mask)

        # Converter o array resultante de volta para uma imagem PIL
        result_image = Image.fromarray(result_array)
        
        new_img = ImageTk.PhotoImage(result_image)
            
        # print(result_image)
        
        self.label.config(image=new_img)
        self.label.image = new_img

        
# criar interface
root = Tk()
interface = processImage("./imagem/eliene.png", root)
interface.show()