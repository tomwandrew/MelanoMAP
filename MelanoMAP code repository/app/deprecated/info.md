This pipeline was deprecated in favour of running each step individually to provide greater control, modularity, and reproducibility. The original workflow automated segmentation, feature extraction, AI-based prediction, and statistical analysis in a single process. However, separating these components allows for:

	-	More rigorous quality control at each stage of the analysis.
	-	Flexible model updates for AI segmentation and classification.
	-	Refinements to survival modelling without re-running the entire pipeline.

The methodology remains unchanged—integrating handcrafted (HC) features with TME-derived predictors and their association with recurrence-free survival (RFS). All steps, including image segmentation, feature extraction, and prognostic modelling (Cox, RSF, DeepSurv), are still implemented, but now as distinct processes to enhance data integrity and interpretability.
