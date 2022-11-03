#ifndef GESTICREGRESSORSVM_H
#define GESTICREGRESSORSVM_H

#include <QObject>
#include <svm.h>
#include <QMessageBox>
#include <QtWidgets>
#include <datasmoothor.h>

#define DEFAULT_PARAM "-t 2 -c 100"

enum SVMFeatures{ SD_SVM, CIC_SVM, SD_DIFF_SVM, CIC_DIFF_SVM, SD_DIFF_NORM_SVM, CIC_DIFF_NORM_SVM };

struct SVMPoint{
    double sd_c0, sd_c1, sd_c2, sd_c3, sd_c4,
           cic_c0, cic_c1, cic_c2, cic_c3, cic_c4,
           sd_diff_0, sd_diff_1, sd_diff_2, sd_diff_3, sd_diff_4, sd_diff_5, sd_diff_6, sd_diff_7, sd_diff_8, sd_diff_9,
           cic_diff_0, cic_diff_1, cic_diff_2, cic_diff_3, cic_diff_4, cic_diff_5, cic_diff_6, cic_diff_7, cic_diff_8, cic_diff_9,
            sd_diff_0_normed, sd_diff_1_normed, sd_diff_2_normed, sd_diff_3_normed, sd_diff_4_normed, sd_diff_5_normed, sd_diff_6_normed, sd_diff_7_normed, sd_diff_8_normed, sd_diff_9_normed,
            cic_diff_0_normed, cic_diff_1_normed, cic_diff_2_normed, cic_diff_3_normed, cic_diff_4_normed, cic_diff_5_normed, cic_diff_6_normed, cic_diff_7_normed, cic_diff_8_normed, cic_diff_9_normed;

    double y;
};

struct NormParam{
    // {{max, min}, {max, min}, ...}
    // for data
    double max_min_data[10][2] = {{std::numeric_limits<double>::min(), std::numeric_limits<double>::max()},
                                {std::numeric_limits<double>::min(), std::numeric_limits<double>::max()},
                                {std::numeric_limits<double>::min(), std::numeric_limits<double>::max()},
                                {std::numeric_limits<double>::min(), std::numeric_limits<double>::max()},
                                {std::numeric_limits<double>::min(), std::numeric_limits<double>::max()},
                                {std::numeric_limits<double>::min(), std::numeric_limits<double>::max()},
                                {std::numeric_limits<double>::min(), std::numeric_limits<double>::max()},
                                {std::numeric_limits<double>::min(), std::numeric_limits<double>::max()},
                                {std::numeric_limits<double>::min(), std::numeric_limits<double>::max()},
                                {std::numeric_limits<double>::min(), std::numeric_limits<double>::max()}};

    // for label
    double max_min_label[2] = {std::numeric_limits<double>::min(), std::numeric_limits<double>::max()};
};

class GesticRegressorSVM : public QWidget
{
    Q_OBJECT
public:
    GesticRegressorSVM(QWidget *parent = nullptr);

    void setDefaultParam();
    void buildSVMProblem();
    void train();
    void evaluate();
    double predict(SVMPoint);
    bool isTrained();

    void setData(QList<SVMPoint>);
    void setFileName(QString);
    void setParam(svm_parameter);
    void setSlidingWindowParam(int, int);

    void release();

    QList<SVMFeatures> getFeatureList() const;
    void setFeatureList(QList<SVMFeatures>);

    void enableSmoother();
    void disableSmoother();

private:
    void saveDataToFile();
    void saveModelToFile();

    void preProcessData();
    void preProcessLabel();
    void updateNormParam(QList<SVMPoint>);
    void normalize(SVMPoint*, bool normalize_label_ = false);
    void denormalize();
    void calGradient(QList<SVMPoint>*, bool);
    void calGradientChannelWise(QList<SVMPoint>*);

    bool loadModelFromFile(QString);
    bool loadNormParamFromFile(QString);
    void saveNormParamToFile();

    void smoothing(SVMPoint *p);
    void resetSmoothors();



private:
    struct svm_parameter _param;		// set by parse_command_line
    struct svm_problem _prob;		// set by read_problem
    struct svm_node *_x_space = nullptr;
    struct svm_model *_svm_model = nullptr;

    QList<SVMPoint> _train_point_list;
    QList<SVMPoint> _last_predict_point_list;   // length-2 list, storaging the last two elements

    QString _model_file_name, _data_file_name, _normparam_file_name;
    bool _is_trained;

    struct NormParam _norm_param;   // normalization parameter for data and label

    double _y_h_max_min[2]; // max and min prediction in the training set
    int _window_size, _step;

    QList<SVMFeatures> _feature_list;

    MovAvg *_mov_avg_sd0, *_mov_avg_sd1, *_mov_avg_sd2, *_mov_avg_sd3, *_mov_avg_sd4,
           *_mov_avg_cic0, *_mov_avg_cic1, *_mov_avg_cic2, *_mov_avg_cic3, *_mov_avg_cic4;

    bool _smoothing = false;

signals:
    void gesticRegressorSVMPredict(double, double*);

public slots:
    void onLoadModelFromFile();

};

#endif // GESTICREGRESSORSVM_H
