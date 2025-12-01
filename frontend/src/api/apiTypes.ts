export interface ModelArchitecture {
  name: string;
  file: string;
  size: string;
  modified: string;
}

export interface TrainedModel {
  name: string;
  display_name: string;
  size: string;
  modified: string;
  folder?: string;
}
