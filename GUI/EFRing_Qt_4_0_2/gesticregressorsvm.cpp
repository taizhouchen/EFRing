#include "gesticregressorsvm.h"

GesticRegressorSVM::GesticRegressorSVM(QWidget *parent) : QWidget(parent)
{
    _is_trained = false;
    _smoothing = false;
    _window_size = 3;
    _step = 1; // do not change
    _feature_list.append(SVMFeatures::SD_DIFF_SVM);
    setDefaultParam();

    _mov_avg_sd0 = new MovAvg(50);
    _mov_avg_sd1 = new MovAvg(50);
    _mov_avg_sd2 = new MovAvg(50);
    _mov_avg_sd3 = new MovAvg(50);
    _mov_avg_sd4 = new MovAvg(50);

    _mov_avg_cic0 = new MovAvg(50);
    _mov_avg_cic1 = new MovAvg(50);
    _mov_avg_cic2 = new MovAvg(50);
    _mov_avg_cic3 = new MovAvg(50);
    _mov_avg_cic4 = new MovAvg(50);
}

void GesticRegressorSVM::setDefaultParam()
{
    // default values
    _param.svm_type = EPSILON_SVR;
    _param.kernel_type = LINEAR;
    _param.degree = 3;
    _param.gamma = 0;	// 1/num_features
    _param.coef0 = 0;
    _param.nu = 0.5;
    _param.cache_size = 100;
    _param.C = 1;
    _param.eps = 1e-3;
    _param.p = 0.1;
    _param.shrinking = 1;
    _param.probability = 0;
    _param.nr_weight = 0;
    _param.weight_label = NULL;
    _param.weight = NULL;
}

void GesticRegressorSVM::buildSVMProblem()
{
    if(_feature_list.isEmpty()) return;

    if(_train_point_list.isEmpty()){
            QMessageBox msgBox;
            msgBox.setIcon(QMessageBox::Warning);
            msgBox.setText("No data to train");
            msgBox.setWindowTitle("Regressor Message");
            msgBox.exec();
            return;
        }

        const char *error_msg;
        error_msg = svm_check_parameter(&_prob,&_param);

        if(error_msg)
        {
            QMessageBox msgBox;
            msgBox.setIcon(QMessageBox::Warning);
            msgBox.setText(tr("ERROR: %1\n").arg(error_msg));
            msgBox.setWindowTitle("Regressor Message");
            msgBox.exec();
            return;
        }

        /*

        [.  .  .  .  .] .  .  .  .  .
         . [.  .  .  .  .] .  .  .  .
         .  . [.  .  .  .  .] .  .  .
         .  .  . [.  .  .  .  .] .  .
         .  .  .  . [.  .  .  .  .] .
         .  .  .  .  . [.  .  .  .  .]

        */
        _prob.l = (_train_point_list.size() - _window_size + 1) / _step;
        _prob.y = new double[_prob.l];

        if(_param.svm_type == EPSILON_SVR ||
                _param.svm_type == NU_SVR ||
                _param.svm_type == C_SVC ||
                _param.svm_type == ONE_CLASS){
            if(_param.gamma == 0) _param.gamma = 1;

            int channel_count = 0;
            for(int i = 0; i < _feature_list.size(); i++){
                switch (_feature_list[i]) {
                    case SVMFeatures::SD_SVM:
                        channel_count += 5;
                    break;
                    case SVMFeatures::SD_DIFF_SVM:
                        channel_count += 10;
                    break;
                    case SVMFeatures::SD_DIFF_NORM_SVM:
                        channel_count += 10;
                    break;

                    case SVMFeatures::CIC_SVM:
                        channel_count += 5;
                    break;
                    case SVMFeatures::CIC_DIFF_SVM:
                        channel_count += 10;
                    break;
                    case SVMFeatures::CIC_DIFF_NORM_SVM:
                        channel_count += 10;
                    break;
                }
            }

            if(channel_count == 0) return;

            _x_space = new svm_node[(channel_count * _window_size + 1) * _prob.l];
            _prob.x = new svm_node *[_prob.l];

            int i = 0;
            for (QList<SVMPoint>::iterator q = _train_point_list.begin() + _window_size - 1; q + _step - 1 != _train_point_list.end(); q+=_step, i++)
            {
                for(int j = 0; j < _window_size; j++){

                    int channel_ = 0;
                    for(int f = 0; f < _feature_list.size(); f++){

                        switch (_feature_list[f]) {

                            case SVMFeatures::SD_SVM:

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_c0;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_c1;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_c2;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_c3;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_c4;
                                channel_++;

                                break;

                            case SVMFeatures::SD_DIFF_SVM:

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_0;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_1;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_2;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_3;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_4;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_5;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_6;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_7;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_8;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_9;
                                channel_++;

                                break;

                            case SVMFeatures::SD_DIFF_NORM_SVM:

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_0_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_1_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_2_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_3_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_4_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_5_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_6_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_7_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_8_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->sd_diff_9_normed;
                                channel_++;

                                break;

                            case SVMFeatures::CIC_SVM:

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_c0;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_c1;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_c2;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_c3;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_c4;
                                channel_++;

                                break;

                            case SVMFeatures::CIC_DIFF_SVM:

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_0;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_1;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_2;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_3;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_4;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_5;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_6;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_7;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_8;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_9;
                                channel_++;

                                break;

                            case SVMFeatures::CIC_DIFF_NORM_SVM:

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_0_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_1_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_2_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_3_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_4_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_5_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_6_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_7_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_8_normed;
                                channel_++;

                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].index = channel_ + 1 + (channel_count * j);
                                _x_space[(channel_count * _window_size + 1) * i + channel_ + (channel_count * j)].value = (q - (_window_size - j - 1))->cic_diff_9_normed;
                                channel_++;

                                break;
                        }

                    }

                }

                _x_space[(channel_count * _window_size + 1) * i + (channel_count *_window_size)].index = -1;
                _prob.x[i] = &_x_space[(channel_count * _window_size + 1) * i];
                _prob.y[i] = q->y;

            }
            saveDataToFile();
            saveNormParamToFile();
        }
}

void GesticRegressorSVM::train(){
    qDebug() << "Training start";
    _svm_model = svm_train(&_prob, &_param);
    _is_trained = true;
    qDebug() << "Training done" ;
    saveModelToFile();
}

void GesticRegressorSVM::evaluate()
{

}


double GesticRegressorSVM::predict(SVMPoint svm_point_x)
{
    double y_h;

    /* smoothing */
    if(_smoothing) smoothing(&svm_point_x);

    /* normalization */
    normalize(&svm_point_x);

    /* sliding windows */
    _last_predict_point_list.push_back(svm_point_x);
    if(_last_predict_point_list.size() < _window_size) return y_h;

//    calGradient(&_last_point_list, true);   // keep the last point for next round

    QList<SVMPoint> _last_predict_point_list_temp = _last_predict_point_list;
    calGradientChannelWise(&_last_predict_point_list_temp);

    int channel_count = 0;
    for(int i = 0; i < _feature_list.size(); i++){
        switch (_feature_list[i]) {
            case SVMFeatures::SD_SVM:
                channel_count += 5;
            break;
            case SVMFeatures::SD_DIFF_SVM:
                channel_count += 10;
            break;
            case SVMFeatures::SD_DIFF_NORM_SVM:
                channel_count += 10;
            break;

            case SVMFeatures::CIC_SVM:
                channel_count += 5;
            break;
            case SVMFeatures::CIC_DIFF_SVM:
                channel_count += 10;
            break;
            case SVMFeatures::CIC_DIFF_NORM_SVM:
                channel_count += 10;
            break;
        }
    }

    if(channel_count == 0) return 0;

    svm_node x[channel_count * _window_size + 1];

    for(int j = 0; j < _window_size; j++){


        int channel_ = 0;
        for(int f = 0; f < _feature_list.size(); f++){

            switch (_feature_list[f]) {
                case SVMFeatures::SD_SVM:

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_c0;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_c1;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_c2;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_c3;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_c4;
                    channel_++;

                    break;

                case SVMFeatures::SD_DIFF_SVM:

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_0;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_1;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_2;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_3;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_4;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_5;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_6;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_7;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_8;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_9;
                    channel_++;

                    break;

                case SVMFeatures::SD_DIFF_NORM_SVM:

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_0_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_1_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_2_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_3_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_4_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_5_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_6_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_7_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_8_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].sd_diff_9_normed;
                    channel_++;

                    break;

                case SVMFeatures::CIC_SVM:

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_c0;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_c1;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_c2;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_c3;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_c4;
                    channel_++;

                    break;

                case SVMFeatures::CIC_DIFF_SVM:

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_0;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_1;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_2;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_3;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_4;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_5;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_6;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_7;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_8;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_9;
                    channel_++;


                    break;

                case SVMFeatures::CIC_DIFF_NORM_SVM:

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_0_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_1_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_2_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_3_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_4_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_5_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_6_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_7_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_8_normed;
                    channel_++;

                    x[channel_ + (j * channel_count)].index = channel_ + 1 + (j * channel_count);
                    x[channel_ + (j * channel_count)].value = _last_predict_point_list_temp[j].cic_diff_9_normed;
                    channel_++;

                    break;
            }
        }
    }

    x[channel_count * _window_size].index = -1;

    y_h = svm_predict(_svm_model, x);

    emit gesticRegressorSVMPredict(y_h, _y_h_max_min);

    for(int i = 0; i < _step; i++)
        _last_predict_point_list.pop_front();

    return y_h;
}

void GesticRegressorSVM::preProcessData()
{
    /* calculate gradient for data and label */
//    calGradient(&_point_list, false);

    /* smoothing */
    if(_smoothing){
        resetSmoothors();
        for(int i = 0; i < _train_point_list.size(); i++)
            smoothing(&_train_point_list[i]);
        resetSmoothors();
    }

    /* normalizing signal */
    updateNormParam(_train_point_list);

    for(int i = 0; i < _train_point_list.size(); i++)
        normalize(&_train_point_list[i]);

    calGradientChannelWise(&_train_point_list);
}

bool GesticRegressorSVM::isTrained()
{
    return _svm_model != nullptr && _is_trained;
}

void GesticRegressorSVM::setData(QList<SVMPoint> point_list)
{
    _train_point_list = point_list;
    preProcessData();
}

void GesticRegressorSVM::setFileName(QString file_name)
{
    _data_file_name = file_name + ".data";
    _model_file_name = file_name + ".model";
    _normparam_file_name = file_name + ".normparam";
}

void GesticRegressorSVM::setParam(svm_parameter param)
{
    this->_param = param;
}

void GesticRegressorSVM::setSlidingWindowParam(int window_size_, int step_)
{
    if(window_size_ <= 0) window_size_ = 1;
    _window_size = window_size_;
    step_ = 1;
    _step = step_;
}

void GesticRegressorSVM::release()
{
    svm_free_and_destroy_model(&_svm_model);
    free(_param.weight_label);
    free(_param.weight);
    delete[] _prob.x;
    delete[] _prob.y;
    delete[] _x_space;
    qDebug() << "Regressor released";
}

void GesticRegressorSVM::saveDataToFile()
{
    if(!_data_file_name.isNull())
    {
        FILE *fp = fopen(_data_file_name.toLatin1().constData(),"w");

        QString str = DEFAULT_PARAM;
        const char *p = str.toLatin1().constData();
        const char* svm_type_str = strstr(p, "-s ");
        int svm_type = EPSILON_SVR;
        if(svm_type_str != NULL)
            sscanf(svm_type_str, "-s %d", &svm_type);

        if(fp)
        {
            if(svm_type == EPSILON_SVR || svm_type == NU_SVR)
            {
                int channel_count = 0;
                for(int i = 0; i < _feature_list.size(); i++){
                    switch (_feature_list[i]) {
                        case SVMFeatures::SD_SVM:
                            channel_count += 5;
                        break;
                        case SVMFeatures::SD_DIFF_SVM:
                            channel_count += 10;
                        break;
                        case SVMFeatures::SD_DIFF_NORM_SVM:
                            channel_count += 10;
                        break;

                        case SVMFeatures::CIC_SVM:
                            channel_count += 5;
                        break;
                        case SVMFeatures::CIC_DIFF_SVM:
                            channel_count += 10;
                        break;
                        case SVMFeatures::CIC_DIFF_NORM_SVM:
                            channel_count += 10;
                        break;
                    }
                }

                for(int i = 0; i < _prob.l; i++){
                    fprintf(fp, "%f", _prob.y[i]);
                    for(int j = 0; j < (channel_count * _window_size); j++){
                        fprintf(fp, " %d:%f", j + 1, _x_space[(channel_count * _window_size + 1) * i + j].value);
                    }
                    fprintf(fp, "\n");
                }
            }
            else
            {

            }
            fclose(fp);
            qDebug() << tr("Data was saved to %1").arg(_data_file_name);
        }
    }
}

void GesticRegressorSVM::saveModelToFile()
{
    if(svm_save_model(_model_file_name.toStdString().c_str(),_svm_model))
    {
        qDebug() << tr("Cannot save model to file %1").arg(_model_file_name);
        return;
    }
    qDebug() << tr("Model was saved to %1").arg(_model_file_name);
}

void GesticRegressorSVM::preProcessLabel()
{

}

void GesticRegressorSVM::updateNormParam(QList<SVMPoint> point_list)
{
    for(int i = 0; i < 10; i++){
        _norm_param.max_min_data[i][0] = std::numeric_limits<double>::min();
        _norm_param.max_min_data[i][1] = std::numeric_limits<double>::max();
    }
    _norm_param.max_min_label[0] = std::numeric_limits<double>::min();
    _norm_param.max_min_label[1] = std::numeric_limits<double>::max();

    for(int i = 0; i < point_list.size(); i++){
        if(point_list[i].sd_c0 > _norm_param.max_min_data[0][0]) _norm_param.max_min_data[0][0] = point_list[i].sd_c0;
        if(point_list[i].sd_c0 < _norm_param.max_min_data[0][1]) _norm_param.max_min_data[0][1] = point_list[i].sd_c0;

        if(point_list[i].sd_c1 > _norm_param.max_min_data[1][0]) _norm_param.max_min_data[1][0] = point_list[i].sd_c1;
        if(point_list[i].sd_c1 < _norm_param.max_min_data[1][1]) _norm_param.max_min_data[1][1] = point_list[i].sd_c1;

        if(point_list[i].sd_c2 > _norm_param.max_min_data[2][0]) _norm_param.max_min_data[2][0] = point_list[i].sd_c2;
        if(point_list[i].sd_c2 < _norm_param.max_min_data[2][1]) _norm_param.max_min_data[2][1] = point_list[i].sd_c2;

        if(point_list[i].sd_c3 > _norm_param.max_min_data[3][0]) _norm_param.max_min_data[3][0] = point_list[i].sd_c3;
        if(point_list[i].sd_c3 < _norm_param.max_min_data[3][1]) _norm_param.max_min_data[3][1] = point_list[i].sd_c3;

        if(point_list[i].sd_c4 > _norm_param.max_min_data[4][0]) _norm_param.max_min_data[4][0] = point_list[i].sd_c4;
        if(point_list[i].sd_c4 < _norm_param.max_min_data[4][1]) _norm_param.max_min_data[4][1] = point_list[i].sd_c4;


        if(point_list[i].cic_c0 > _norm_param.max_min_data[5][0]) _norm_param.max_min_data[5][0] = point_list[i].cic_c0;
        if(point_list[i].cic_c0 < _norm_param.max_min_data[5][1]) _norm_param.max_min_data[5][1] = point_list[i].cic_c0;

        if(point_list[i].cic_c1 > _norm_param.max_min_data[6][0]) _norm_param.max_min_data[6][0] = point_list[i].cic_c1;
        if(point_list[i].cic_c1 < _norm_param.max_min_data[6][1]) _norm_param.max_min_data[6][1] = point_list[i].cic_c1;

        if(point_list[i].cic_c2 > _norm_param.max_min_data[7][0]) _norm_param.max_min_data[7][0] = point_list[i].cic_c2;
        if(point_list[i].cic_c2 < _norm_param.max_min_data[7][1]) _norm_param.max_min_data[7][1] = point_list[i].cic_c2;

        if(point_list[i].cic_c3 > _norm_param.max_min_data[8][0]) _norm_param.max_min_data[8][0] = point_list[i].cic_c3;
        if(point_list[i].cic_c3 < _norm_param.max_min_data[8][1]) _norm_param.max_min_data[8][1] = point_list[i].cic_c3;

        if(point_list[i].cic_c4 > _norm_param.max_min_data[9][0]) _norm_param.max_min_data[9][0] = point_list[i].cic_c4;
        if(point_list[i].cic_c4 < _norm_param.max_min_data[9][1]) _norm_param.max_min_data[9][1] = point_list[i].cic_c4;


        if(point_list[i].y > _norm_param.max_min_label[0]) _norm_param.max_min_label[0] = point_list[i].y;
        if(point_list[i].y < _norm_param.max_min_label[1]) _norm_param.max_min_label[1] = point_list[i].y;
    }
}

void GesticRegressorSVM::normalize(SVMPoint *p, bool normalize_label_)
{
    p->sd_c0 = (p->sd_c0 - _norm_param.max_min_data[0][1]) / ((_norm_param.max_min_data[0][0] - _norm_param.max_min_data[0][1]) + 1e-10);
    p->sd_c1 = (p->sd_c1 - _norm_param.max_min_data[1][1]) / ((_norm_param.max_min_data[1][0] - _norm_param.max_min_data[1][1]) + 1e-10);
    p->sd_c2 = (p->sd_c2 - _norm_param.max_min_data[2][1]) / ((_norm_param.max_min_data[2][0] - _norm_param.max_min_data[2][1]) + 1e-10);
    p->sd_c3 = (p->sd_c3 - _norm_param.max_min_data[3][1]) / ((_norm_param.max_min_data[3][0] - _norm_param.max_min_data[3][1]) + 1e-10);
    p->sd_c4 = (p->sd_c4 - _norm_param.max_min_data[4][1]) / ((_norm_param.max_min_data[4][0] - _norm_param.max_min_data[4][1]) + 1e-10);

    p->cic_c0 = (p->cic_c0 - _norm_param.max_min_data[5][1]) / ((_norm_param.max_min_data[5][0] - _norm_param.max_min_data[5][1]) + 1e-10);
    p->cic_c1 = (p->cic_c1 - _norm_param.max_min_data[6][1]) / ((_norm_param.max_min_data[6][0] - _norm_param.max_min_data[6][1]) + 1e-10);
    p->cic_c2 = (p->cic_c2 - _norm_param.max_min_data[7][1]) / ((_norm_param.max_min_data[7][0] - _norm_param.max_min_data[7][1]) + 1e-10);
    p->cic_c3 = (p->cic_c3 - _norm_param.max_min_data[8][1]) / ((_norm_param.max_min_data[8][0] - _norm_param.max_min_data[8][1]) + 1e-10);
    p->cic_c4 = (p->cic_c4 - _norm_param.max_min_data[9][1]) / ((_norm_param.max_min_data[9][0] - _norm_param.max_min_data[9][1]) + 1e-10);

    if(normalize_label_)
        p->y = (p->y - _norm_param.max_min_label[1]) / ((_norm_param.max_min_label[0] - _norm_param.max_min_label[1]) + 1e-10);
}

void GesticRegressorSVM::calGradient(QList<SVMPoint> *point_list, bool keep_last)
{
    for(int i = 0; i < point_list->size()-1; i++){
        (*point_list)[i].sd_c0 = (*point_list)[i+1].sd_c0 - (*point_list)[i].sd_c0;
        (*point_list)[i].sd_c1 = (*point_list)[i+1].sd_c1 - (*point_list)[i].sd_c1;
        (*point_list)[i].sd_c2 = (*point_list)[i+1].sd_c2 - (*point_list)[i].sd_c2;
        (*point_list)[i].sd_c3 = (*point_list)[i+1].sd_c3 - (*point_list)[i].sd_c3;
        (*point_list)[i].sd_c4 = (*point_list)[i+1].sd_c4 - (*point_list)[i].sd_c4;

        (*point_list)[i].cic_c0 = (*point_list)[i+1].cic_c0 - (*point_list)[i].cic_c0;
        (*point_list)[i].cic_c1 = (*point_list)[i+1].cic_c1 - (*point_list)[i].cic_c1;
        (*point_list)[i].cic_c2 = (*point_list)[i+1].cic_c2 - (*point_list)[i].cic_c2;
        (*point_list)[i].cic_c3 = (*point_list)[i+1].cic_c3 - (*point_list)[i].cic_c3;
        (*point_list)[i].cic_c4 = (*point_list)[i+1].cic_c4 - (*point_list)[i].cic_c4;

        (*point_list)[i].y = (*point_list)[i+1].y - (*point_list)[i].y;
    }

    if(!keep_last)
        point_list->pop_back();
}

void GesticRegressorSVM::calGradientChannelWise(QList<SVMPoint> *point_list)
{
    for(int i = 0; i < point_list->size(); i++){

        /*
         [0, 0, 0, 0, 1, 1, 1, 2, 2, 3]
         [1, 2, 3, 4, 2, 3, 4, 3, 4, 4]
        */

        (*point_list)[i].sd_diff_0 = (*point_list)[i].sd_c1 - (*point_list)[i].sd_c0;
        (*point_list)[i].sd_diff_1 = (*point_list)[i].sd_c2 - (*point_list)[i].sd_c0;
        (*point_list)[i].sd_diff_2 = (*point_list)[i].sd_c3 - (*point_list)[i].sd_c0;
        (*point_list)[i].sd_diff_3 = (*point_list)[i].sd_c4 - (*point_list)[i].sd_c0;
        (*point_list)[i].sd_diff_4 = (*point_list)[i].sd_c2 - (*point_list)[i].sd_c1;
        (*point_list)[i].sd_diff_5 = (*point_list)[i].sd_c3 - (*point_list)[i].sd_c1;
        (*point_list)[i].sd_diff_6 = (*point_list)[i].sd_c4 - (*point_list)[i].sd_c1;
        (*point_list)[i].sd_diff_7 = (*point_list)[i].sd_c3 - (*point_list)[i].sd_c2;
        (*point_list)[i].sd_diff_8 = (*point_list)[i].sd_c4 - (*point_list)[i].sd_c2;
        (*point_list)[i].sd_diff_9 = (*point_list)[i].sd_c4 - (*point_list)[i].sd_c3;


        (*point_list)[i].cic_diff_0 = (*point_list)[i].cic_c1 - (*point_list)[i].cic_c0;
        (*point_list)[i].cic_diff_1 = (*point_list)[i].cic_c2 - (*point_list)[i].cic_c0;
        (*point_list)[i].cic_diff_2 = (*point_list)[i].cic_c3 - (*point_list)[i].cic_c0;
        (*point_list)[i].cic_diff_3 = (*point_list)[i].cic_c4 - (*point_list)[i].cic_c0;
        (*point_list)[i].cic_diff_4 = (*point_list)[i].cic_c2 - (*point_list)[i].cic_c1;
        (*point_list)[i].cic_diff_5 = (*point_list)[i].cic_c3 - (*point_list)[i].cic_c1;
        (*point_list)[i].cic_diff_6 = (*point_list)[i].cic_c4 - (*point_list)[i].cic_c1;
        (*point_list)[i].cic_diff_7 = (*point_list)[i].cic_c3 - (*point_list)[i].cic_c2;
        (*point_list)[i].cic_diff_8 = (*point_list)[i].cic_c4 - (*point_list)[i].cic_c2;
        (*point_list)[i].cic_diff_9 = (*point_list)[i].cic_c4 - (*point_list)[i].cic_c3;

    }

    for(int i = 0; i < point_list->size(); i++){

        // normalize sd diff
        double max = std::numeric_limits<double>::min(), min = std::numeric_limits<double>::max();
        if((*point_list)[i].sd_diff_0 > max) max = (*point_list)[i].sd_diff_0;
        if((*point_list)[i].sd_diff_0 < min) min = (*point_list)[i].sd_diff_0;

        if((*point_list)[i].sd_diff_1 > max) max = (*point_list)[i].sd_diff_1;
        if((*point_list)[i].sd_diff_1 < min) min = (*point_list)[i].sd_diff_1;

        if((*point_list)[i].sd_diff_2 > max) max = (*point_list)[i].sd_diff_2;
        if((*point_list)[i].sd_diff_2 < min) min = (*point_list)[i].sd_diff_2;

        if((*point_list)[i].sd_diff_3 > max) max = (*point_list)[i].sd_diff_3;
        if((*point_list)[i].sd_diff_3 < min) min = (*point_list)[i].sd_diff_3;

        if((*point_list)[i].sd_diff_4 > max) max = (*point_list)[i].sd_diff_4;
        if((*point_list)[i].sd_diff_4 < min) min = (*point_list)[i].sd_diff_4;

        if((*point_list)[i].sd_diff_5 > max) max = (*point_list)[i].sd_diff_5;
        if((*point_list)[i].sd_diff_5 < min) min = (*point_list)[i].sd_diff_5;

        if((*point_list)[i].sd_diff_6 > max) max = (*point_list)[i].sd_diff_6;
        if((*point_list)[i].sd_diff_6 < min) min = (*point_list)[i].sd_diff_6;

        if((*point_list)[i].sd_diff_7 > max) max = (*point_list)[i].sd_diff_7;
        if((*point_list)[i].sd_diff_7 < min) min = (*point_list)[i].sd_diff_7;

        if((*point_list)[i].sd_diff_8 > max) max = (*point_list)[i].sd_diff_8;
        if((*point_list)[i].sd_diff_8 < min) min = (*point_list)[i].sd_diff_8;

        if((*point_list)[i].sd_diff_9 > max) max = (*point_list)[i].sd_diff_9;
        if((*point_list)[i].sd_diff_9 < min) min = (*point_list)[i].sd_diff_9;

        (*point_list)[i].sd_diff_0_normed = ((*point_list)[i].sd_diff_0 - min) / (max - min + 1e-10);
        (*point_list)[i].sd_diff_1_normed = ((*point_list)[i].sd_diff_1 - min) / (max - min + 1e-10);
        (*point_list)[i].sd_diff_2_normed = ((*point_list)[i].sd_diff_2 - min) / (max - min + 1e-10);
        (*point_list)[i].sd_diff_3_normed = ((*point_list)[i].sd_diff_3 - min) / (max - min + 1e-10);
        (*point_list)[i].sd_diff_4_normed = ((*point_list)[i].sd_diff_4 - min) / (max - min + 1e-10);
        (*point_list)[i].sd_diff_5_normed = ((*point_list)[i].sd_diff_5 - min) / (max - min + 1e-10);
        (*point_list)[i].sd_diff_6_normed = ((*point_list)[i].sd_diff_6 - min) / (max - min + 1e-10);
        (*point_list)[i].sd_diff_7_normed = ((*point_list)[i].sd_diff_7 - min) / (max - min + 1e-10);
        (*point_list)[i].sd_diff_8_normed = ((*point_list)[i].sd_diff_8 - min) / (max - min + 1e-10);
        (*point_list)[i].sd_diff_9_normed = ((*point_list)[i].sd_diff_9 - min) / (max - min + 1e-10);


        // normalize cic diff
        max = std::numeric_limits<double>::min();
        min = std::numeric_limits<double>::max();

        if((*point_list)[i].cic_diff_0 > max) max = (*point_list)[i].cic_diff_0;
        if((*point_list)[i].cic_diff_0 < min) min = (*point_list)[i].cic_diff_0;

        if((*point_list)[i].cic_diff_1 > max) max = (*point_list)[i].cic_diff_1;
        if((*point_list)[i].cic_diff_1 < min) min = (*point_list)[i].cic_diff_1;

        if((*point_list)[i].cic_diff_2 > max) max = (*point_list)[i].cic_diff_2;
        if((*point_list)[i].cic_diff_2 < min) min = (*point_list)[i].cic_diff_2;

        if((*point_list)[i].cic_diff_3 > max) max = (*point_list)[i].cic_diff_3;
        if((*point_list)[i].cic_diff_3 < min) min = (*point_list)[i].cic_diff_3;

        if((*point_list)[i].cic_diff_4 > max) max = (*point_list)[i].cic_diff_4;
        if((*point_list)[i].cic_diff_4 < min) min = (*point_list)[i].cic_diff_4;

        if((*point_list)[i].cic_diff_5 > max) max = (*point_list)[i].cic_diff_5;
        if((*point_list)[i].cic_diff_5 < min) min = (*point_list)[i].cic_diff_5;

        if((*point_list)[i].cic_diff_6 > max) max = (*point_list)[i].cic_diff_6;
        if((*point_list)[i].cic_diff_6 < min) min = (*point_list)[i].cic_diff_6;

        if((*point_list)[i].cic_diff_7 > max) max = (*point_list)[i].cic_diff_7;
        if((*point_list)[i].cic_diff_7 < min) min = (*point_list)[i].cic_diff_7;

        if((*point_list)[i].cic_diff_8 > max) max = (*point_list)[i].cic_diff_8;
        if((*point_list)[i].cic_diff_8 < min) min = (*point_list)[i].cic_diff_8;

        if((*point_list)[i].cic_diff_9 > max) max = (*point_list)[i].cic_diff_9;
        if((*point_list)[i].cic_diff_9 < min) min = (*point_list)[i].cic_diff_9;

        (*point_list)[i].cic_diff_0_normed = ((*point_list)[i].cic_diff_0 - min) / (max - min + 1e-10);
        (*point_list)[i].cic_diff_1_normed = ((*point_list)[i].cic_diff_1 - min) / (max - min + 1e-10);
        (*point_list)[i].cic_diff_2_normed = ((*point_list)[i].cic_diff_2 - min) / (max - min + 1e-10);
        (*point_list)[i].cic_diff_3_normed = ((*point_list)[i].cic_diff_3 - min) / (max - min + 1e-10);
        (*point_list)[i].cic_diff_4_normed = ((*point_list)[i].cic_diff_4 - min) / (max - min + 1e-10);
        (*point_list)[i].cic_diff_5_normed = ((*point_list)[i].cic_diff_5 - min) / (max - min + 1e-10);
        (*point_list)[i].cic_diff_6_normed = ((*point_list)[i].cic_diff_6 - min) / (max - min + 1e-10);
        (*point_list)[i].cic_diff_7_normed = ((*point_list)[i].cic_diff_7 - min) / (max - min + 1e-10);
        (*point_list)[i].cic_diff_8_normed = ((*point_list)[i].cic_diff_8 - min) / (max - min + 1e-10);
        (*point_list)[i].cic_diff_9_normed = ((*point_list)[i].cic_diff_9 - min) / (max - min + 1e-10);
    }
}

bool GesticRegressorSVM::loadModelFromFile(QString file_name)
{
    _svm_model = svm_load_model(file_name.toStdString().c_str());

    if(_svm_model == 0){
        QMessageBox msgBox;
        msgBox.setIcon(QMessageBox::Critical);
        msgBox.setText(QString("Invalid file %1").arg(file_name));
        msgBox.setWindowTitle("Message");
        msgBox.exec();

        _svm_model = nullptr;
        _is_trained = false;

        return false;
    }


    return true;
}

bool GesticRegressorSVM::loadNormParamFromFile(QString file_name)
{
    QFile file(file_name);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)){
            QMessageBox msgBox;
            msgBox.setIcon(QMessageBox::Critical);
            msgBox.setText(QString("Failed to load .normParam file %1").arg(file_name));
            msgBox.setWindowTitle("Message");
            msgBox.exec();

           return false;
     }

    QTextStream in(&file);
    int i = 0;
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList split = line.split(' ');
        _norm_param.max_min_data[i][0] = split[0].toDouble();
        _norm_param.max_min_data[i][1] = split[1].toDouble();
        i++;
    }

    return true;
}

void GesticRegressorSVM::saveNormParamToFile()
{
    FILE *fp = fopen(_normparam_file_name.toLatin1().constData(),"w");

    if(fp){
        for(int i = 0; i < 10; i++){
            fprintf(fp, "%f %f\n", _norm_param.max_min_data[i][0], _norm_param.max_min_data[i][1]);
        }
        fclose(fp);
        qDebug() << tr("Norm param was saved to %1").arg(_normparam_file_name);
    }
}

void GesticRegressorSVM::smoothing(SVMPoint *p)
{
    _mov_avg_sd0->update(&p->sd_c0);
    _mov_avg_sd1->update(&p->sd_c1);
    _mov_avg_sd2->update(&p->sd_c2);
    _mov_avg_sd3->update(&p->sd_c3);
    _mov_avg_sd4->update(&p->sd_c4);

    _mov_avg_cic0->update(&p->cic_c0);
    _mov_avg_cic1->update(&p->cic_c1);
    _mov_avg_cic2->update(&p->cic_c2);
    _mov_avg_cic3->update(&p->cic_c3);
    _mov_avg_cic4->update(&p->cic_c4);
}

void GesticRegressorSVM::resetSmoothors()
{
    _mov_avg_sd0->reset();
    _mov_avg_sd1->reset();
    _mov_avg_sd2->reset();
    _mov_avg_sd3->reset();
    _mov_avg_sd4->reset();

    _mov_avg_cic0->reset();
    _mov_avg_cic1->reset();
    _mov_avg_cic2->reset();
    _mov_avg_cic3->reset();
    _mov_avg_cic4->reset();
}

QList<SVMFeatures> GesticRegressorSVM::getFeatureList() const
{
    return _feature_list;
}

void GesticRegressorSVM::setFeatureList(QList<SVMFeatures> feature_list_)
{
    _feature_list = feature_list_;
}

void GesticRegressorSVM::enableSmoother()
{
    _smoothing = true;
}

void GesticRegressorSVM::disableSmoother()
{
    _smoothing = false;
}

void GesticRegressorSVM::onLoadModelFromFile()
{
    QString file_name = QFileDialog::getOpenFileName(this, tr("Load Model"));
    if(file_name != ""){
        if(loadModelFromFile(file_name))
            if(loadNormParamFromFile(file_name.split('.')[0] + ".normparam"))
                _is_trained = true;
    }
}
