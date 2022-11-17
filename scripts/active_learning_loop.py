def loop():
    """pseudocode
    Input: Unlabeled dataset Du, Base model Mb, Acquisition Batch Size B,
           Strategy S, Labeling budget L
    Output: Labeled dataset Dl, Trained model Mt

    Query oracle for initial seed dataset Ds from Du (not RANDOM!!)
    Let Dl = Ds
    Mt = Train(Mb, Dl)
    while |Dl| < L do
        Ds = SelectInformativeSamples(Du , Mt , S, B)
        D's = Pre-tag(Ds, Mt)
        D''s = Query oracle for labels to D's
        Move new labeled instances D''s from Du to Dl
        Mt = Train(Mb, Dl)
    return Dl, Mt
    """
    pass


def main():
    # constants, dataset loading, model instantiation, etc.
    loop()


if __name__ == "__main__":
    main()
