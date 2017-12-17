
# Change to a non-source folder to make sure we run the tests on the
# installed library.
- "cd C:\\"

$installed_skcycling_folder = $(python -c "import os; os.chdir('c:/'); import skcycling;\
print(os.path.dirname(skcycling.__file__))")
echo "skcycling found in: $installed_skcycling_folder"

# --pyargs argument is used to make sure we run the tests on the
# installed package rather than on the local folder
py.test --pyargs skcycling $installed_skcycling_folder
exit $LastExitCode
