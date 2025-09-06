import os
import io
import glob
import lzma
import tarfile
import datasets


class Openwebtext(datasets.GeneratorBasedBuilder):
    """The Open WebText dataset (direct .xz loader)."""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            description="Plain text",
            version=datasets.Version("1.0.0"),
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="An open-source replication of the WebText dataset from OpenAI.",
            features=datasets.Features({"text": datasets.Value("string")}),
            homepage="https://skylion007.github.io/OpenWebTextCorpus/",
            citation="@misc{Gokaslan2019OpenWeb, title={OpenWebText Corpus}, author={Aaron Gokaslan*, Vanya Cohen*, Ellie Pavlick, Stefanie Tellex}, year={2019}}",
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.manual_dir or dl_manager.download_and_extract(self.config.data_dir)
        files = glob.glob(os.path.join(data_dir, "*.xz"))
        if not files:
            raise FileNotFoundError(f"No .xz files found in {data_dir}")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files": files},
            ),
        ]

    def _generate_examples(self, files):
        """Yields examples from .xz files directly."""
        for filepath in files:
            if not filepath.endswith(".xz"):
                continue

            # Open the .xz file as a binary stream
            with lzma.open(filepath, "rb") as compressed_file:
                # Open the tar archive directly from the streaming decompressor
                with tarfile.open(fileobj=compressed_file, mode="r") as tar:
                    # Iterate through each member (file) in the tar archive
                    for member in tar.getmembers():
                        # Read the content of the file
                        if member.isfile():
                            f = tar.extractfile(member)
                            text = f.read().decode('utf-8')
                            
                            # Process and yield the extracted text
                            yield f"{filepath}/{member.name}", {"text": text}