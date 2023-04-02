import gradio as gr
from main import NST,image_loader
def main(input1,input2): # Style and content
    imsize = 512
    style_lyr = ['conv_1','conv_2','conv_3','conv_4','conv_5']
    content_lyr = ['conv_5']
    style_img,content_img,input_img = image_loader(input1,input2,"noise",imsize,True)
    out = NST(content_lyr,content_img,style_lyr,style_img,input_img,500,1000000,1,True)
    return out
    # pass# Implement your sketch recognition model here...

# Define the inputs
input1 = gr.inputs.Image(label="Input 1",type="pil")

input2 = gr.inputs.Image(label="Input 2",type="pil")

# Define the output
output = gr.outputs.Image(label="Output",type="pil")

# Create the interface
interface = gr.Interface(main, inputs=[input1, input2], outputs=output)

# Launch the interface
interface.launch()
