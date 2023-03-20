from typing import List, Dict, Union
from fastapi import FastAPI
from pydantic import BaseModel
from model import MLModelHandler, DLModelHandler
import uvicorn

app = FastAPI()

# assign model handler as global variable [2 LINES]
ml_handler = MLModelHandler()
dl_handler = DLModelHandler()


# define request/response data type for validation
class RequestModel(BaseModel):
    text: Union[str, List[str]]
    model_type: str

class ResponseModel(BaseModel):
    # prediction: {"label":"negative", "score":0.9752}
    #dictionary(json)파일 형식으로 반환하겠다.
    prediction: Dict


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict", response_model=ResponseModel)
async def predict(request: RequestModel):
    text = request.text
    text = [text] if isinstance(text, str) else text
    model_type = request.model_type

    # model inference [2 LINES]
    if model_type == 'ml':
        predictions = ml_handler.handle(text)
    else:
        predictions = dl_handler.handle(text)

    # response
    result = {str(i): {'text': t, 'label': l, 'confidence': c}
                         for i, (t, l, c) in enumerate(zip(text, predictions[0], predictions[1]))}

    return ResponseModel(prediction=result)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)

#웹에서 사용하려면 http://localhost:8000/docs or http://localhost:8000/