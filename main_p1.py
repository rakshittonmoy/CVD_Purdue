"""Implements alignment algorithm."""

import argparse
from alignment_model_assignment_2 import AlignmentModel_2
# from alignment_model_assignment_2 import AlignmentModel_1


def main():
  """Main function to run the alignment model."""
  parser = argparse.ArgumentParser(description='CS59300CVD Assignment 1')
  parser.add_argument('-i', '--image_name', required=True,
                      type=str, help='Input image path')
  parser.add_argument('-m', '--metric', default='ncc',
                      type=str, help='Metric to use for alignment')
  args = parser.parse_args()
  print(args.image_name)
  if args.image_name == 'all':
    run_all()
    return

  model = AlignmentModel_2(args.image_name, metric=args.metric)
  model.align()
  model.save('%s_%s_aligned.png' %
             (args.image_name.split('.')[0], args.metric))


def run_all():
  """Run the alignment model on all images."""
  # Assignment 1
  # for metric in ['ncc', 'mse', 'ssim']:
  #   for image_name in range(1, 7):
  #     image_name = 'data/%d.jpg' % image_name
  #     model = AlignmentModel_1(image_name, metric=metric)
  #     model.align()
  #     model.save('%s_%s_aligned.png' % (image_name.split('.')[0], metric))

  # Assignment 2
  for metric in ['ncc', 'mse']:
    for image_name in range(1, 7):
      image_name = 'data/%d.jpg' % image_name
      model = AlignmentModel_2(image_name, metric=metric)
      model.align()
      model.save('%s_%s_aligned.png' % (image_name.split('.')[0], metric))


if __name__ == '__main__':
  main()
