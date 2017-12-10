# runs python files sequentially to generate final solution

cd preprocessing
python create_user_data.py
python create_product_data.py
python create_aisle_data.py
python create_department_data.py

cd ../rnn_product
python prepare_product_data.py
python rnn_product.py
python rnn_product_bmm.py

cd ../rnn_aisle
python prepare_aisle_data.py
python rnn_aisle.py

cd ../rnn_department
python prepare_department_data.py
python rnn_department.py

cd ../blend
python prepare_blend_data.py
python gbm_blend.py
python nn_blend.py
python submit.py
