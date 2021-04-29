for option in units_num activation optimizer
do
    echo "python -u testcases.py  --h_param=$option"
    python -u testcases.py  --h_param=$option
done
