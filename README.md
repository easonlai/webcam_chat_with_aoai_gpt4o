# Chat with GPT-4o with Webcam

I trust everyone can't wait to start building something with a fantastic [GPT-4o](https://openai.com/index/hello-gpt-4o/) multimodal model from [OpenAI](https://openai.com/). At the [Microsoft Build 2024](https://build.microsoft.com/) event, Microsoft [announced the general availability of GPT-4o](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/announcing-multimodal-innovations-in-generative-ai-with-azure/ba-p/4146804) with text and image capability. I tried to quickly build something to make traditional conversational chat more fun by adding a camera so the model could see.

This is a simple prototype code repository sharing with the basic web UI. I used simple components such as [Flask](https://pypi.org/project/Flask/), [CV2 (OpenCV)](https://pypi.org/project/opencv-python/), [OpenAI client](https://pypi.org/project/openai/), and the rest of the simple baseline libraries to make this. You need to create a deployment of [GPT-4o](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-4o-and-gpt-4-turbo) in [Azure OpenAI Services](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview).

This program logic is simple: the webpage will ask permission to turn on the camera. It would automatically take a snapshot of the image every second. Then, you can type in some text in the chat windows, and it will post the text together with the image to the GPT-4o multimodel model at the same time and then get a response. So you can ask some questions like what you saw.

The response from GPT-4o is not fast at the moment, but it works, and I look forward to the service adding audio and video support soon. 

![alt text](https://github.com/easonlai/webcam_chat_with_aoai_gpt4o/blob/main/git-images/git-image-1.png)

Enjoy!

 ### Prerequisites

- Python 3.6+
- Flask
- OpenAI
- cv2
- numpy

### Installing & Make it Run

1. Clone the repository:
    ```
    git clone https://github.com/easonlai/webcam_chat_with_aoai_gpt4o.git
    ```
2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Run the application:
    ```
    python run.py
    ```

## Usage

1. Navigate to the home page (`localhost:5000` by default).
2. Give access to the camera, then show and ask it something.
3. GPT-4o will process the image and the text at the same time and give the result.

### Let's have some fun

![alt text](https://github.com/easonlai/webcam_chat_with_aoai_gpt4o/blob/main/git-images/git-image-2.png)

![alt text](https://github.com/easonlai/webcam_chat_with_aoai_gpt4o/blob/main/git-images/git-image-3.png)

![alt text](https://github.com/easonlai/webcam_chat_with_aoai_gpt4o/blob/main/git-images/git-image-4.png)

![alt text](https://github.com/easonlai/webcam_chat_with_aoai_gpt4o/blob/main/git-images/git-image-5.png)

