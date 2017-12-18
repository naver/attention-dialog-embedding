# DBDC3 of DSTC6
## NCDS Team

## Attention-based Dialog Embedding for Dialog Breakdown Detection
Despite the recent advent of dialog systems, we have seen that there are still many challenges unresolved and the system often generates responses causing a breakdown in the interaction between a user and the system.
The dialog breakdown significantly damages user experience, and thus detecting such failure is significant.
We propose a model to detect dialog breakdown using the recurrent neural network and attention layer to embed a previous dialog context. 
This model determines the probability of breakdown using the extracted dialog context vector and the target sentence's representation vector.
We submitted this study to the [Dialog Breakdown Detection Challenge 3](https://dbd-challenge.github.io/dbdc3/) of [Dialog System Technology Challenge 6](http://workshop.colips.org/dstc6/), and the results showed that it significantly outperforms the most of other models in estimating breakdown probability.

### Installing dependencies
Recommend you to use *virtualenv*!
 `pip3 install -r requirements.txt`
You need pytorch, torchwordemb and more. Please refer to the `requirements.txt` file.

### before running the main file
1. you need data files in ./data directory
- glove twitter vectors
- [dataset](https://github.com/dbd-challenge/dbdc3) provided by DBDC 3 
Note that you need to modify the given data into the proper format for our model
Automatic conversion code will be provided soon.
2. check the parser options


### run main_glove file
`python3 main_glove.py`

### License
This project is distributed under Apache License, Version 2.0.
See [LICENSE](LICENSE) for full license text.

```
Copyright 2017 NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```


